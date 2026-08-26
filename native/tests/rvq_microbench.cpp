#include <arm_neon.h>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr std::size_t rows = 4224, groups = 48, stages = 2;
constexpr std::size_t padded_rows = (rows + 63) & ~std::size_t{63};
constexpr std::size_t chunks = padded_rows / 64;

void current8(const float* lookup, const std::uint8_t* indices, float* output) {
  for (std::size_t r = 0; r < rows; r += 8) {
    float sums[8]{};
    std::size_t chunk[8], lane[8]; bool high[8];
    for (std::size_t i=0;i<8;++i) { chunk[i]=(r+i)/64; lane[i]=(r+i)%32; high[i]=((r+i)%64)>=32; }
    for (std::size_t s=0;s<stages;++s) for (std::size_t g=0;g<groups;++g) {
      const float* values=lookup+(s*groups+g)*16;
      for (std::size_t i=0;i<8;++i) {
        const auto packed=indices[((s*chunks+chunk[i])*groups+g)*32+lane[i]];
        sums[i]+=values[high[i] ? packed>>4 : packed&15];
      }
    }
    std::memcpy(output+r,sums,std::min<std::size_t>(8,rows-r)*sizeof(float));
  }
}

void current8_four(const float* lookup,const std::uint8_t* indices,float* output){
  for(std::size_t token=0;token<4;++token)current8(lookup+token*stages*groups*16,indices,output+token*rows);
}

void batch4(const float* lookup,const std::uint8_t* indices,float* output){
  constexpr std::size_t lookup_size=stages*groups*16;
  for(std::size_t r=0;r<rows;r+=8){
    float sums[4][8]{};
    std::size_t chunk[8],lane[8];bool high[8];
    for(std::size_t i=0;i<8;++i){chunk[i]=(r+i)/64;lane[i]=(r+i)%32;high[i]=((r+i)%64)>=32;}
    for(std::size_t s=0;s<stages;++s)for(std::size_t g=0;g<groups;++g){
      std::uint8_t codes[8];
      for(std::size_t i=0;i<8;++i){const auto packed=indices[((s*chunks+chunk[i])*groups+g)*32+lane[i]];codes[i]=static_cast<std::uint8_t>(high[i]?packed>>4:packed&15);}
      for(std::size_t token=0;token<4;++token){const float* values=lookup+token*lookup_size+(s*groups+g)*16;for(std::size_t i=0;i<8;++i)sums[token][i]+=values[codes[i]];}
    }
    for(std::size_t token=0;token<4;++token)std::memcpy(output+token*rows+r,sums[token],std::min<std::size_t>(8,rows-r)*sizeof(float));
  }
}

void unrolled16(const float* lookup, const std::uint8_t* indices, float* output) {
  for (std::size_t r=0;r<rows;r+=16) {
    float sums[16]{};
    for (std::size_t s=0;s<stages;++s) for (std::size_t g=0;g<groups;++g) {
      const float* values=lookup+(s*groups+g)*16;
      for (std::size_t i=0;i<16 && r+i<rows;++i) {
        const std::size_t row=r+i, chunk=row/64, lane=row%32;
        const auto packed=indices[((s*chunks+chunk)*groups+g)*32+lane];
        sums[i]+=values[(row%64)>=32 ? packed>>4 : packed&15];
      }
    }
    std::memcpy(output+r,sums,std::min<std::size_t>(16,rows-r)*sizeof(float));
  }
}

void tbl16(const float* lookup, const std::uint8_t* indices, float* output) {
  for (std::size_t r=0;r<rows;r+=16) {
    float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
    for (std::size_t s=0;s<stages;++s) for (std::size_t g=0;g<groups;++g) {
      alignas(16) std::uint8_t codes[16];
      for (std::size_t i=0;i<16;++i) {
        const std::size_t row=r+i, chunk=row/64, lane=row%32;
        const auto packed=indices[((s*chunks+chunk)*groups+g)*32+lane];
        codes[i]=static_cast<std::uint8_t>((row%64)>=32 ? packed>>4 : packed&15);
      }
      const uint8x16_t selector=vld1q_u8(codes);
      std::array<uint8x16_t,4> planes{};
      for (std::size_t byte=0;byte<4;++byte) {
        alignas(16) std::uint8_t table[16];
        for (std::size_t code=0;code<16;++code) {
          const auto bits=std::bit_cast<std::uint32_t>(lookup[(s*groups+g)*16+code]);
          table[code]=static_cast<std::uint8_t>(bits>>(byte*8));
        }
        planes[byte]=vqtbl1q_u8(vld1q_u8(table),selector);
      }
      const uint32x4_t lo0=vreinterpretq_u32_u8(vzip1q_u8(planes[0],planes[1]));
      const uint32x4_t hi0=vreinterpretq_u32_u8(vzip1q_u8(planes[2],planes[3]));
      const uint32x4_t lo1=vreinterpretq_u32_u8(vzip2q_u8(planes[0],planes[1]));
      const uint32x4_t hi1=vreinterpretq_u32_u8(vzip2q_u8(planes[2],planes[3]));
      const uint16x8_t first=vzip1q_u16(vreinterpretq_u16_u32(lo0),vreinterpretq_u16_u32(hi0));
      const uint16x8_t second=vzip2q_u16(vreinterpretq_u16_u32(lo0),vreinterpretq_u16_u32(hi0));
      const uint16x8_t third=vzip1q_u16(vreinterpretq_u16_u32(lo1),vreinterpretq_u16_u32(hi1));
      const uint16x8_t fourth=vzip2q_u16(vreinterpretq_u16_u32(lo1),vreinterpretq_u16_u32(hi1));
      a=vaddq_f32(a,vreinterpretq_f32_u16(first)); b=vaddq_f32(b,vreinterpretq_f32_u16(second));
      c=vaddq_f32(c,vreinterpretq_f32_u16(third)); d=vaddq_f32(d,vreinterpretq_f32_u16(fourth));
    }
    alignas(16) float sums[16]; vst1q_f32(sums,a);vst1q_f32(sums+4,b);vst1q_f32(sums+8,c);vst1q_f32(sums+12,d);
    std::memcpy(output+r,sums,std::min<std::size_t>(16,rows-r)*sizeof(float));
  }
}

template<class F> double measure(F&& fn,const float* l,const std::uint8_t* i,float* o) {
  for(int n=0;n<10;++n) fn(l,i,o);
  const auto start=std::chrono::steady_clock::now();
  for(int n=0;n<200;++n) fn(l,i,o);
  return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/200;
}
}

int main(){
  std::mt19937 rng(250); std::uniform_real_distribution<float> fd(-4,4);
  std::vector<float> lookup(stages*groups*16), reference(rows), candidate(rows);
  std::vector<std::uint8_t> indices(stages*chunks*groups*32);
  for(auto& value:lookup)value=fd(rng); for(auto& value:indices)value=static_cast<std::uint8_t>(rng());
  current8(lookup.data(),indices.data(),reference.data());
  auto check=[&](auto fn,const char* name){ fn(lookup.data(),indices.data(),candidate.data());
    const bool equal=std::memcmp(reference.data(),candidate.data(),rows*sizeof(float))==0;
    const double us=measure(fn,lookup.data(),indices.data(),candidate.data());
    std::cout<<name<<" equal="<<(equal?"true":"false")<<" us="<<std::fixed<<std::setprecision(3)<<us<<'\n';
    if(!equal) std::exit(1); return us; };
  const double base=check(current8,"current8"); const double u16=check(unrolled16,"unrolled16");
  const double table=check(tbl16,"tbl16");
  std::cout<<"unrolled16_gain="<<(base/u16-1)*100<<"% tbl16_gain="<<(base/table-1)*100<<"%\n";
  std::vector<float> batch_lookup(4*stages*groups*16),batch_reference(4*rows),batch_candidate(4*rows);for(auto& value:batch_lookup)value=fd(rng);
  current8_four(batch_lookup.data(),indices.data(),batch_reference.data());batch4(batch_lookup.data(),indices.data(),batch_candidate.data());
  const bool batch_equal=std::memcmp(batch_reference.data(),batch_candidate.data(),4*rows*sizeof(float))==0;
  const double four_us=measure(current8_four,batch_lookup.data(),indices.data(),batch_reference.data());
  const double batch_us=measure(batch4,batch_lookup.data(),indices.data(),batch_candidate.data());
  std::cout<<"batch4 equal="<<(batch_equal?"true":"false")<<" four_calls_us="<<four_us<<" batch4_us="<<batch_us<<" gain="<<(four_us/batch_us-1)*100<<"%\n";
  if(!batch_equal)return 1;
}
