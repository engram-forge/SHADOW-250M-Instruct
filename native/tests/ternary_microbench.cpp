#include <arm_neon.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr std::size_t rows=4224, columns=1536, stride=(columns+4)/5;
std::size_t packed_offset(std::size_t row,std::size_t block){return ((row/8)*stride+block)*8+row%8;}
std::size_t nibble_offset(std::size_t row,std::size_t column){return ((row/16)*columns+column)*8;}
std::size_t bitplane_offset(std::size_t row,std::size_t column){return ((row/16)*columns+column)*4;}

const auto& digits(){ static const auto value=[] {
  std::array<std::array<std::int8_t,5>,256> table{};
  for(std::size_t byte=0;byte<256;++byte){unsigned v=byte;for(auto& d:table[byte]){d=static_cast<std::int8_t>(v%3)-1;v/=3;}}
  return table;}(); return value; }

void expanded16(const float* x,const std::uint8_t* expanded,float* y){
  for(std::size_t r=0;r<rows;r+=16){
    float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
    for(std::size_t col=0;col<columns;++col){
      const int8x8_t bytes=vreinterpret_s8_u8(vld1_u8(expanded+nibble_offset(r,col)));
      const int8x16_t code=vcombine_s8(vshr_n_s8(vshl_n_s8(bytes,4),4),vshr_n_s8(bytes,4));
      const int16x8_t lo=vmovl_s8(vget_low_s8(code)),hi=vmovl_s8(vget_high_s8(code));
      a=vfmaq_n_f32(a,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),x[col]);
      b=vfmaq_n_f32(b,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),x[col]);
      c=vfmaq_n_f32(c,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),x[col]);
      d=vfmaq_n_f32(d,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),x[col]);
    }
    vst1q_f32(y+r,a);vst1q_f32(y+r+4,b);vst1q_f32(y+r+8,c);vst1q_f32(y+r+12,d);
  }
}

void bitplane16(const float* x,const std::uint8_t* planes,float* y){
  const int16x8_t shifts={0,-1,-2,-3,-4,-5,-6,-7};
  for(std::size_t r=0;r<rows;r+=16){
    float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
    for(std::size_t col=0;col<columns;++col){
      const auto* p=planes+bitplane_offset(r,col);
      const std::uint16_t positive=static_cast<std::uint16_t>(p[0]|(p[1]<<8));
      const std::uint16_t negative=static_cast<std::uint16_t>(p[2]|(p[3]<<8));
      const uint16x8_t one=vdupq_n_u16(1);
      const uint16x8_t pl=vandq_u16(vshlq_u16(vdupq_n_u16(positive),shifts),one);
      const uint16x8_t nl=vandq_u16(vshlq_u16(vdupq_n_u16(negative),shifts),one);
      const uint16x8_t ph=vandq_u16(vshlq_u16(vdupq_n_u16(positive>>8),shifts),one);
      const uint16x8_t nh=vandq_u16(vshlq_u16(vdupq_n_u16(negative>>8),shifts),one);
      const int16x8_t lo=vsubq_s16(vreinterpretq_s16_u16(pl),vreinterpretq_s16_u16(nl));
      const int16x8_t hi=vsubq_s16(vreinterpretq_s16_u16(ph),vreinterpretq_s16_u16(nh));
      a=vfmaq_n_f32(a,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),x[col]);
      b=vfmaq_n_f32(b,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),x[col]);
      c=vfmaq_n_f32(c,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),x[col]);
      d=vfmaq_n_f32(d,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),x[col]);
    }
    vst1q_f32(y+r,a);vst1q_f32(y+r+4,b);vst1q_f32(y+r+8,c);vst1q_f32(y+r+12,d);
  }
}

void expanded16_batch4(const float* x,const std::uint8_t* expanded,float* y){
  for(std::size_t r=0;r<rows;r+=16){
    float32x4_t sums[4][4];
    for(auto& token:sums)for(auto& lane:token)lane=vdupq_n_f32(0);
    for(std::size_t col=0;col<columns;++col){
      const int8x8_t bytes=vreinterpret_s8_u8(vld1_u8(expanded+nibble_offset(r,col)));
      const int8x16_t code=vcombine_s8(vshr_n_s8(vshl_n_s8(bytes,4),4),vshr_n_s8(bytes,4));
      const int16x8_t lo=vmovl_s8(vget_low_s8(code)),hi=vmovl_s8(vget_high_s8(code));
      const float32x4_t weights[4]={vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)))};
      for(std::size_t token=0;token<4;++token)for(std::size_t lane=0;lane<4;++lane)
        sums[token][lane]=vfmaq_n_f32(sums[token][lane],weights[lane],x[token*columns+col]);
    }
    for(std::size_t token=0;token<4;++token)for(std::size_t lane=0;lane<4;++lane)
      vst1q_f32(y+token*rows+r+lane*4,sums[token][lane]);
  }
}

void expanded16_four_calls(const float* x,const std::uint8_t* expanded,float* y){
  for(std::size_t token=0;token<4;++token)expanded16(x+token*columns,expanded,y+token*rows);
}
void paired_four_calls(const float* x,const std::uint8_t* a,const std::uint8_t* b,float* ya,float* yb){expanded16_four_calls(x,a,ya);expanded16_four_calls(x,b,yb);}
void paired_batch4(const float* x,const std::uint8_t* a,const std::uint8_t* b,float* ya,float* yb){expanded16_batch4(x,a,ya);expanded16_batch4(x,b,yb);}

void expanded16_i8(const float* x,const std::int8_t* expanded,float* y){
  for(std::size_t r=0;r<rows;r+=16){
    float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
    const std::int8_t* block=expanded+(r/16)*columns*16;
    for(std::size_t col=0;col<columns;++col){
      const int8x16_t code=vld1q_s8(block+col*16);
      const int16x8_t lo=vmovl_s8(vget_low_s8(code)),hi=vmovl_s8(vget_high_s8(code));
      a=vfmaq_n_f32(a,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),x[col]);
      b=vfmaq_n_f32(b,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),x[col]);
      c=vfmaq_n_f32(c,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),x[col]);
      d=vfmaq_n_f32(d,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),x[col]);
    }
    vst1q_f32(y+r,a);vst1q_f32(y+r+4,b);vst1q_f32(y+r+8,c);vst1q_f32(y+r+12,d);
  }
}

void expanded16_mask(const float* x,const std::uint8_t* expanded,float* y){
  for(std::size_t r=0;r<rows;r+=16){
    float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
    for(std::size_t col=0;col<columns;++col){
      const int8x8_t bytes=vreinterpret_s8_u8(vld1_u8(expanded+nibble_offset(r,col)));
      const int8x16_t code=vcombine_s8(vshr_n_s8(vshl_n_s8(bytes,4),4),vshr_n_s8(bytes,4));
      const float32x4_t plus=vdupq_n_f32(x[col]),minus=vnegq_f32(plus),z=vdupq_n_f32(0);
      auto select=[&](int8x8_t lanes,bool high){
        const int16x8_t wide=vmovl_s8(lanes);
        const int32x4_t values=high?vmovl_s16(vget_high_s16(wide)):vmovl_s16(vget_low_s16(wide));
        const uint32x4_t pm=vcgtq_s32(values,vdupq_n_s32(0));
        const uint32x4_t nm=vcltq_s32(values,vdupq_n_s32(0));
        return vbslq_f32(pm,plus,vbslq_f32(nm,minus,z));
      };
      a=vaddq_f32(a,select(vget_low_s8(code),false));
      b=vaddq_f32(b,select(vget_low_s8(code),true));
      c=vaddq_f32(c,select(vget_high_s8(code),false));
      d=vaddq_f32(d,select(vget_high_s8(code),true));
    }
    vst1q_f32(y+r,a);vst1q_f32(y+r+4,b);vst1q_f32(y+r+8,c);vst1q_f32(y+r+12,d);
  }
}

template<std::size_t Tile, std::size_t Prefetch>
void expanded16_tiled(const float* x,const std::uint8_t* expanded,float* y){
  for(std::size_t r=0;r<rows;r+=16){
    float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
    for(std::size_t tile=0;tile<columns;tile+=Tile){
      const std::size_t tile_end=std::min(columns,tile+Tile);
      for(std::size_t col=tile;col<tile_end;++col){
        if constexpr(Prefetch!=0)
          if(col+Prefetch<tile_end)__builtin_prefetch(expanded+nibble_offset(r,col+Prefetch),0,1);
        const int8x8_t bytes=vreinterpret_s8_u8(vld1_u8(expanded+nibble_offset(r,col)));
        const int8x16_t code=vcombine_s8(vshr_n_s8(vshl_n_s8(bytes,4),4),vshr_n_s8(bytes,4));
        const int16x8_t lo=vmovl_s8(vget_low_s8(code)),hi=vmovl_s8(vget_high_s8(code));
        a=vfmaq_n_f32(a,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),x[col]);
        b=vfmaq_n_f32(b,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),x[col]);
        c=vfmaq_n_f32(c,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),x[col]);
        d=vfmaq_n_f32(d,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),x[col]);
      }
    }
    vst1q_f32(y+r,a);vst1q_f32(y+r+4,b);vst1q_f32(y+r+8,c);vst1q_f32(y+r+12,d);
  }
}

void compact8_tbl(const float* x,const std::uint8_t* packed,float* y){
  alignas(16) static std::array<std::array<std::uint8_t,256>,5> unsigned_digits=[] {
    std::array<std::array<std::uint8_t,256>,5> result{};
    for(std::size_t j=0;j<5;++j)for(std::size_t i=0;i<256;++i)result[j][i]=static_cast<std::uint8_t>(digits()[i][j]);
    return result;}();
  for(std::size_t r=0;r<rows;r+=8){
    float32x4_t lo=vdupq_n_f32(0),hi=vdupq_n_f32(0);
    for(std::size_t block=0;block<stride;++block){
      const uint8x8_t codes=vld1_u8(packed+packed_offset(r,block));
      for(std::size_t j=0;j<5 && block*5+j<columns;++j){
        // ARMv8 TBL tables are at most 64 bytes, so select the 64-entry quarter first.
        const uint8x8_t quarter=vshr_n_u8(codes,6), index=vand_u8(codes,vdup_n_u8(63));
        uint8x8_t selected=vdup_n_u8(0);
        for(std::uint8_t q=0;q<4;++q){
          const uint8x16x4_t table={{vld1q_u8(unsigned_digits[j].data()+q*64),vld1q_u8(unsigned_digits[j].data()+q*64+16),vld1q_u8(unsigned_digits[j].data()+q*64+32),vld1q_u8(unsigned_digits[j].data()+q*64+48)}};
          const uint8x8_t value=vqtbl4_u8(table,index);
          selected=vbsl_u8(vceq_u8(quarter,vdup_n_u8(q)),value,selected);
        }
        const int16x8_t wide=vmovl_s8(vreinterpret_s8_u8(selected));
        lo=vfmaq_n_f32(lo,vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))),x[block*5+j]);
        hi=vfmaq_n_f32(hi,vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide))),x[block*5+j]);
      }
    }
    vst1q_f32(y+r,lo);vst1q_f32(y+r+4,hi);
  }
}
template<class F,class W> double measure(F fn,const float* x,const W* w,float* y){for(int i=0;i<5;++i)fn(x,w,y);auto s=std::chrono::steady_clock::now();for(int i=0;i<100;++i)fn(x,w,y);return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-s).count()/100;}
}
int main(){
  std::mt19937 rng(250);std::uniform_real_distribution<float> f(-2,2);
  std::vector<float>x(columns),reference(rows),candidate(rows);for(auto&v:x)v=f(rng);
  std::vector<float>batch_x(4*columns),batch_reference(4*rows),batch_candidate(4*rows);for(auto&v:batch_x)v=f(rng);
  std::vector<std::uint8_t> packed(rows*stride),expanded(rows/2*columns),bitplanes(rows/4*columns);
  std::vector<std::int8_t> expanded_i8(rows*columns);
  for(std::size_t r=0;r<rows;++r)for(std::size_t b=0;b<stride;++b)packed[packed_offset(r,b)]=static_cast<std::uint8_t>(rng()%243);
  for(std::size_t r=0;r<rows;r+=16)for(std::size_t c=0;c<columns;++c){auto* p=expanded.data()+nibble_offset(r,c);auto* bp=bitplanes.data()+bitplane_offset(r,c);auto* p8=expanded_i8.data()+(r/16)*columns*16+c*16;std::uint16_t positive=0,negative=0;for(std::size_t lane=0;lane<16;++lane){auto value=digits()[packed[packed_offset(r+lane,c/5)]][c%5];auto code=static_cast<std::uint8_t>(value)&15;p8[lane]=value;if(value>0)positive|=static_cast<std::uint16_t>(1u<<lane);else if(value<0)negative|=static_cast<std::uint16_t>(1u<<lane);if(lane<8)p[lane]=code;else p[lane-8]|=code<<4;}bp[0]=positive&255;bp[1]=positive>>8;bp[2]=negative&255;bp[3]=negative>>8;}
  expanded16(x.data(),expanded.data(),reference.data());
  bitplane16(x.data(),bitplanes.data(),candidate.data());bool equal_bitplane=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  expanded16_tiled<64,0>(x.data(),expanded.data(),candidate.data());bool equal64=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  expanded16_tiled<128,0>(x.data(),expanded.data(),candidate.data());bool equal128=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  expanded16_tiled<128,32>(x.data(),expanded.data(),candidate.data());bool equal_prefetch=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  expanded16_i8(x.data(),expanded_i8.data(),candidate.data());bool equal_i8=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  expanded16_mask(x.data(),expanded.data(),candidate.data());bool equal_mask=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  compact8_tbl(x.data(),packed.data(),candidate.data());bool equal_compact=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
  expanded16_four_calls(batch_x.data(),expanded.data(),batch_reference.data());expanded16_batch4(batch_x.data(),expanded.data(),batch_candidate.data());bool equal_batch4=std::memcmp(batch_reference.data(),batch_candidate.data(),4*rows*sizeof(float))==0;
  const double base=measure(expanded16,x.data(),expanded.data(),reference.data());
  const double bitplane=measure(bitplane16,x.data(),bitplanes.data(),candidate.data());
  const double tile64=measure(expanded16_tiled<64,0>,x.data(),expanded.data(),candidate.data());
  const double tile128=measure(expanded16_tiled<128,0>,x.data(),expanded.data(),candidate.data());
  const double prefetch=measure(expanded16_tiled<128,32>,x.data(),expanded.data(),candidate.data());
  const double i8=measure(expanded16_i8,x.data(),expanded_i8.data(),candidate.data());
  const double mask=measure(expanded16_mask,x.data(),expanded.data(),candidate.data());
  const double compact=measure(compact8_tbl,x.data(),packed.data(),candidate.data());
  const double four_calls=measure(expanded16_four_calls,batch_x.data(),expanded.data(),batch_reference.data());
  const double batch4=measure(expanded16_batch4,batch_x.data(),expanded.data(),batch_candidate.data());
  std::vector<std::uint8_t> expanded_b=expanded;for(auto& byte:expanded_b)byte^=0x11;std::vector<float> pair_a(4*rows),pair_b(4*rows);
  for(int i=0;i<5;++i)paired_four_calls(batch_x.data(),expanded.data(),expanded_b.data(),pair_a.data(),pair_b.data());auto ps=std::chrono::steady_clock::now();for(int i=0;i<100;++i)paired_four_calls(batch_x.data(),expanded.data(),expanded_b.data(),pair_a.data(),pair_b.data());const double paired_calls=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-ps).count()/100;
  for(int i=0;i<5;++i)paired_batch4(batch_x.data(),expanded.data(),expanded_b.data(),pair_a.data(),pair_b.data());ps=std::chrono::steady_clock::now();for(int i=0;i<100;++i)paired_batch4(batch_x.data(),expanded.data(),expanded_b.data(),pair_a.data(),pair_b.data());const double paired_batch=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-ps).count()/100;
  std::cout<<std::boolalpha<<"equal_bitplane="<<equal_bitplane<<" equal64="<<equal64<<" equal128="<<equal128<<" equal_prefetch="<<equal_prefetch<<" equal_i8="<<equal_i8<<" equal_mask="<<equal_mask<<" equal_compact="<<equal_compact<<" equal_batch4="<<equal_batch4
           <<" expanded16_us="<<std::fixed<<std::setprecision(3)<<base<<" bitplane16_us="<<bitplane<<" bitplane_gain="<<(base/bitplane-1)*100<<"% tile64_us="<<tile64<<" tile128_us="<<tile128
           <<" tile128_prefetch32_us="<<prefetch<<" expanded16_i8_us="<<i8<<" mask_us="<<mask<<" compact8_tbl_us="<<compact
           <<" four_calls_us="<<four_calls<<" batch4_us="<<batch4<<" batch4_gain="<<(four_calls/batch4-1)*100<<"%"
           <<" paired_four_us="<<paired_calls<<" paired_batch4_us="<<paired_batch<<" paired_gain="<<(paired_calls/paired_batch-1)*100<<"%"
           <<" nibble_mib="<<(expanded.size()/1048576.0)<<" bitplane_mib="<<(bitplanes.size()/1048576.0)<<" i8_mib="<<(expanded_i8.size()/1048576.0)<<"\n";
  return equal_bitplane&&equal64&&equal128&&equal_prefetch&&equal_i8&&equal_mask&&equal_compact&&equal_batch4?0:1;
}
