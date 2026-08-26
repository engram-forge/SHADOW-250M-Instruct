#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

struct Shape { std::string_view name; std::size_t rows, cols; };
std::size_t fp_off(std::size_t r,std::size_t c,std::size_t n){return ((r/16)*n+c)*16+r%16;}
std::size_t dp_off(std::size_t r,std::size_t c,std::size_t n){return ((r/16)*(n/4)+c/4)*64+(r%16)*4+c%4;}

void fp32(const float*x,const int8_t*w,float*y,Shape s){
  for(size_t r=0;r<s.rows;r+=16){float32x4_t a[4];for(auto&v:a)v=vdupq_n_f32(0);
    for(size_t c=0;c<s.cols;++c){auto q=vld1q_s8(w+fp_off(r,c,s.cols));auto l=vmovl_s8(vget_low_s8(q)),h=vmovl_s8(vget_high_s8(q));
      a[0]=vfmaq_n_f32(a[0],vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))),x[c]);a[1]=vfmaq_n_f32(a[1],vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))),x[c]);
      a[2]=vfmaq_n_f32(a[2],vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),x[c]);a[3]=vfmaq_n_f32(a[3],vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))),x[c]);}
    for(int i=0;i<4;++i)vst1q_f32(y+r+i*4,a[i]);}}

void quant(const float*x,int8_t*q,float*sc,size_t n,size_t group){for(size_t b=0;b<n;b+=group){size_t e=std::min(b+group,n);float p=0;for(size_t c=b;c<e;++c)p=std::max(p,std::abs(x[c]));float s=p/127,iv=s?1/s:0;sc[b/group]=s;for(size_t c=b;c<e;++c)q[c]=int8_t(std::clamp(std::nearbyint(x[c]*iv),-127.f,127.f));}}

#if defined(SHADOW_ARM_DOTPROD)
void dot(const int8_t*x,const int8_t*w,const float*sc,float*y,Shape s,size_t group){
  for(size_t r=0;r<s.rows;r+=16){float32x4_t out[4];for(auto&v:out)v=vdupq_n_f32(0);
    for(size_t b=0;b<s.cols;b+=group){int32x4_t a[4];for(auto&v:a)v=vdupq_n_s32(0);size_t e=std::min(b+group,s.cols);
      for(size_t c=b;c<e;c+=4){int32_t word;__builtin_memcpy(&word,x+c,4);auto xv=vreinterpretq_s8_s32(vdupq_n_s32(word));auto*p=w+dp_off(r,c,s.cols);
        for(int i=0;i<4;++i)a[i]=vdotq_s32(a[i],vld1q_s8(p+i*16),xv);}
      for(int i=0;i<4;++i)out[i]=vfmaq_n_f32(out[i],vcvtq_f32_s32(a[i]),sc[b/group]);}
    for(int i=0;i<4;++i)vst1q_f32(y+r+i*4,out[i]);}}
#endif

template<class F>double bench(F f){for(int i=0;i<3;++i)f();auto t=std::chrono::steady_clock::now();for(int i=0;i<30;++i)f();return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/30;}

int main(){
#if !defined(SHADOW_ARM_DOTPROD)
  std::cout<<"dotprod_unavailable=true\n";return 0;
#else
  Shape ss[]={{"up_gt",4224,1536},{"dn",1536,4224}};std::mt19937 g(250);std::uniform_real_distribution<float>d(-2,2);std::cout<<std::fixed<<std::setprecision(6);
  for(auto s:ss){std::vector<float>x(s.cols),ref(s.rows),got(s.rows);std::vector<int8_t>wf(s.rows*s.cols),wd(s.rows*s.cols),q(s.cols);for(auto&v:x)v=d(g);
    for(size_t r=0;r<s.rows;++r)for(size_t c=0;c<s.cols;++c){auto v=int8_t(g()%3-1);wf[fp_off(r,c,s.cols)]=v;wd[dp_off(r,c,s.cols)]=v;}fp32(x.data(),wf.data(),ref.data(),s);double tf=bench([&]{fp32(x.data(),wf.data(),ref.data(),s);});
    for(size_t group:{s.cols,size_t(128),size_t(64)}){std::vector<float>sc((s.cols+group-1)/group);quant(x.data(),q.data(),sc.data(),s.cols,group);dot(q.data(),wd.data(),sc.data(),got.data(),s,group);double e=0,z=0,m=0;for(size_t i=0;i<s.rows;++i){double v=got[i]-ref[i];e+=v*v;z+=double(ref[i])*ref[i];m=std::max(m,std::abs(v));}double td=bench([&]{quant(x.data(),q.data(),sc.data(),s.cols,group);dot(q.data(),wd.data(),sc.data(),got.data(),s,group);});
      std::cout<<s.name<<" group="<<group<<" fp32_us="<<tf<<" dotprod_total_us="<<td<<" speedup="<<tf/td<<" max_abs="<<m<<" relative_rmse="<<std::sqrt(e/z)<<" weight_mib="<<wd.size()/1048576.0<<'\n';}}
#endif
}
