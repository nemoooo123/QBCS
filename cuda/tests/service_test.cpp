#include "evaluation.h"
#include "qbcs.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fake final : qbcs::BatchEvaluator {
    int calls=0, mode;
    explicit Fake(int m):mode(m) {}
    int capacity() const override { return 7; }
    std::string description() const override { return "fake"; }
    std::string statistics() const override { return "calls="+std::to_string(calls); }
    void evaluate(const std::vector<const qbcs::Sample*>& s,std::vector<int>& f) override {
        if (mode && ++calls==3) {
            if(mode==1) throw std::runtime_error("injected backend failure");
            f.clear(); return;
        }
        if(!mode)++calls;
        require(!s.empty() && s.size()<=7, "batch capacity was violated");
        f.resize(s.size());
        for(size_t i=0;i<s.size();++i)f[i]=s[i]->l1[0];
    }
};

int main() {
    for(int mode=0;mode<3;++mode) {
        qbcs::EvaluationService service(std::make_unique<Fake>(mode),100,2);
        std::atomic<int> failures{0},completed{0},started{0};
        std::vector<std::thread> threads;
        for(int t=0;t<24;++t)threads.emplace_back([&,t]{
            ++started;
            while(started.load()!=24)std::this_thread::yield();
            try {
                for(int it=0;it<100;++it) {
                    const int n=1+(t*13+it)%37;
                    std::vector<qbcs::Sample> s(n); std::vector<int> f(n,-1);
                    for(int i=0;i<n;++i)s[i].l1.push_back((t+it+i)%256);
                    service.evaluate(s.data(),s.size(),f.data());
                    for(int i=0;i<n;++i)require(f[i]==(t+it+i)%256,"result ordering/data mismatch");
                    ++completed;
                }
            } catch(const std::exception&) {++failures;}
        });
        for(auto& thread:threads)thread.join();
        require(mode ? failures==24 : failures==0 && completed==2400,
                "concurrent completion/failure propagation mismatch");
        if(mode) {
            qbcs::Sample s;s.l1.push_back(1);int f=0; bool threw=false;
            try {service.evaluate(&s,1,&f);}catch(const std::exception&){threw=true;}
            require(threw,"new requests must observe an earlier backend failure");
        }
        std::cout<<"mode="<<mode<<"; caught="<<failures<<"; "<<service.statistics()<<"\n";
    }
}
