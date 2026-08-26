#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

enum class WaitMode { spin, wait, hybrid };

class Pool {
public:
  Pool(std::size_t threads, WaitMode mode) : threads_(threads), mode_(mode) {
    for (std::size_t worker=1;worker<threads_;++worker)
      workers_.emplace_back([this,worker]{loop(worker);});
  }
  ~Pool(){stopping_.store(true,std::memory_order_release);generation_.fetch_add(1,std::memory_order_release);generation_.notify_all();for(auto& w:workers_)w.join();}
  void run(std::size_t rows,std::size_t work){
    rows_=rows;work_=work;completed_.store(0,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);generation_.notify_all();
    task(0,boundary(1));
    while(completed_.load(std::memory_order_acquire)!=threads_-1)__asm__ volatile("yield");
  }
  std::uint64_t checksum()const{return checksum_.load(std::memory_order_relaxed);}
private:
  std::size_t boundary(std::size_t worker)const{if(worker==0)return 0;if(worker>=threads_)return rows_;const auto raw=rows_*worker/threads_;return std::min(rows_,(raw+15)&~std::size_t{15});}
  void pause(std::uint64_t seen){
    if(mode_==WaitMode::spin){while(generation_.load(std::memory_order_acquire)==seen)__asm__ volatile("yield");return;}
    if(mode_==WaitMode::hybrid)for(int i=0;i<64;++i){if(generation_.load(std::memory_order_acquire)!=seen)return;__asm__ volatile("yield");}
    generation_.wait(seen,std::memory_order_acquire);
  }
  void task(std::size_t begin,std::size_t end){std::uint64_t sum=0x9e3779b97f4a7c15ULL+begin;for(std::size_t row=begin;row<end;++row)for(std::size_t i=0;i<work_;++i)sum=(sum^(row+i+0x9e37))*0x100000001b3ULL;checksum_.fetch_xor(sum,std::memory_order_relaxed);}
  void loop(std::size_t worker){std::uint64_t seen=0;for(;;){pause(seen);const auto current=generation_.load(std::memory_order_acquire);if(stopping_.load(std::memory_order_acquire))return;task(boundary(worker),boundary(worker+1));completed_.fetch_add(1,std::memory_order_release);seen=current;}}
  std::size_t threads_,rows_=0,work_=0;WaitMode mode_;std::atomic<bool> stopping_{false};std::atomic<std::uint64_t> generation_{0},checksum_{0};std::atomic<std::size_t> completed_{0};std::vector<std::thread> workers_;
};

double measure(WaitMode mode,std::size_t rows,std::size_t work,std::size_t iterations){Pool pool(4,mode);for(int i=0;i<20;++i)pool.run(rows,work);const auto start=std::chrono::steady_clock::now();for(std::size_t i=0;i<iterations;++i)pool.run(rows,work);const auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();if(pool.checksum()==0x1234)std::cerr<<"impossible";return elapsed/iterations;}
const char* name(WaitMode mode){return mode==WaitMode::spin?"spin":mode==WaitMode::wait?"wait":"hybrid";}
int main(){struct Case{std::string_view name;std::size_t rows,work,iterations;};const Case cases[]={{"kv",128,32,2000},{"q",1536,32,1000},{"ternary",4224,64,500},{"logits",131072,2,300}};std::cout<<std::fixed<<std::setprecision(3);for(const auto& test:cases){std::cout<<test.name;for(auto mode:{WaitMode::spin,WaitMode::wait,WaitMode::hybrid})std::cout<<' '<<name(mode)<<"_us="<<measure(mode,test.rows,test.work,test.iterations);std::cout<<'\n';}}
