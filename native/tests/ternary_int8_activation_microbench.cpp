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
struct Shape{std::string_view name;std::size_t rows,cols;};
std::size_t off(std::size_t r,std::size_t c,std::size_t n){return ((r/16)*n+c)*16;}
void fp(const float*x,const int8_t*w,float*y,Shape s){for(size_t r=0;r<s.rows;r+=16){float32x4_t a[4]={};for(auto&v:a)v=vdupq_n_f32(0);for(size_t c=0;c<s.cols;++c){auto q=vld1q_s8(w+off(r,c,s.cols));auto l=vmovl_s8(vget_low_s8(q)),h=vmovl_s8(vget_high_s8(q));a[0]=vfmaq_n_f32(a[0],vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))),x[c]);a[1]=vfmaq_n_f32(a[1],vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))),x[c]);a[2]=vfmaq_n_f32(a[2],vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),x[c]);a[3]=vfmaq_n_f32(a[3],vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))),x[c]);}for(int i=0;i<4;++i)vst1q_f32(y+r+i*4,a[i]);}}
float quant(const float*x,int8_t*q,size_t n){float p=0;for(size_t i=0;i<n;++i)p=std::max(p,std::abs(x[i]));float s=p/127,iv=s?1/s:0;for(size_t i=0;i<n;++i)q[i]=int8_t(std::clamp(std::nearbyint(x[i]*iv),-127.f,127.f));return s;}
void iq(const int8_t*x,const int8_t*w,float scale,float*y,Shape s){for(size_t r=0;r<s.rows;r+=16){int32x4_t a[4]={};for(auto&v:a)v=vdupq_n_s32(0);for(size_t c=0;c<s.cols;++c){auto z=vld1q_s8(w+off(r,c,s.cols));auto xv=vdup_n_s8(x[c]);auto l=vmull_s8(vget_low_s8(z),xv),h=vmull_s8(vget_high_s8(z),xv);a[0]=vaddw_s16(a[0],vget_low_s16(l));a[1]=vaddw_s16(a[1],vget_high_s16(l));a[2]=vaddw_s16(a[2],vget_low_s16(h));a[3]=vaddw_s16(a[3],vget_high_s16(h));}for(int i=0;i<4;++i)vst1q_f32(y+r+i*4,vmulq_n_f32(vcvtq_f32_s32(a[i]),scale));}}
template<class F>double bench(F f){for(int i=0;i<3;++i)f();auto t=std::chrono::steady_clock::now();for(int i=0;i<30;++i)f();return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/30;}
int main(){Shape ss[]={{"up_gt",4224,1536},{"dn",1536,4224}};std::mt19937 g(250);std::uniform_real_distribution<float>d(-2,2);for(auto s:ss){std::vector<float>x(s.cols),a(s.rows),b(s.rows);std::vector<int8_t>q(s.cols),w(s.rows*s.cols);for(auto&v:x)v=d(g);for(auto&v:w)v=int8_t(g()%3-1);fp(x.data(),w.data(),a.data(),s);float sc=quant(x.data(),q.data(),s.cols);iq(q.data(),w.data(),sc,b.data(),s);double e=0,m=0,z=0;for(size_t i=0;i<s.rows;++i){double v=b[i]-a[i];e+=v*v;z+=double(a[i])*a[i];m=std::max(m,std::abs(v));}double tf=bench([&]{fp(x.data(),w.data(),a.data(),s);});double ti=bench([&]{auto v=quant(x.data(),q.data(),s.cols);iq(q.data(),w.data(),v,b.data(),s);});std::cout<<s.name<<" fp32_us="<<tf<<" int8_total_us="<<ti<<" speedup="<<tf/ti<<" max_abs="<<m<<" relative_rmse="<<std::sqrt(e/z)<<"\n";}}
