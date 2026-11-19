#include "Motion.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace moving_tank;
int main(int argc,char** argv){
    bool pass=true;int checks=0;
    std::ofstream log;
    if(argc==2){log.open(argv[1]);log<<std::setprecision(17)<<"scenario,dt_s,time_s,x_command_m,x_measured_m,v_command_m_s,v_measured_m_s,a_command_m_s2,a_measured_m_s2,x_error_m,v_error_m_s,a_error_m_s2,locked_DOF_error\n";}
    auto check=[&](bool ok,const char* what){++checks;if(!ok){pass=false;std::cerr<<"FAIL "<<what<<"\n";}};
    for(auto name:{"STATIC","ACCELERATION","ACCELERATE_STOP","SINUSOIDAL"}){
        Motion m;m.scenario=name;MotionFunction fn(m);std::unique_ptr<chrono::ChFunction> clone(fn.Clone());
        double max_der=0,max_der2=0,max_jump=0;
        for(int i=1;i<2400;++i){double t=i*.00005,e=1e-7;auto s=m.at(t);
            max_der=std::max(max_der,std::abs((m.at(t+e).x-m.at(t-e).x)/(2*e)-s.v));
            max_der2=std::max(max_der2,std::abs((m.at(t+e).v-m.at(t-e).v)/(2*e)-s.a));
            check(clone->GetVal(t)==fn.GetVal(t) && clone->GetDer(t)==fn.GetDer(t) && clone->GetDer2(t)==fn.GetDer2(t),"Clone");
        }
        for(double t:{m.ramp,m.ramp+m.hold,2*m.ramp+m.hold,2*m.ramp+m.hold+m.stop,m.envelope}){
            auto l=m.at(t-1e-10),r=m.at(t+1e-10);max_jump=std::max({max_jump,std::abs(l.x-r.x),std::abs(l.v-r.v),std::abs(l.a-r.a)});
        }
        check(max_der<1e-8 && max_der2<1e-6 && max_jump<1e-7,"analytic derivative / join continuity");
        if(m.scenario=="ACCELERATE_STOP"){auto s=m.at(.12);check(std::abs(s.x-.0002)<1e-15&&s.v==0&&s.a==0,"stop endpoint");}
        for(double dt:{.00005,.000025}){
            chrono::ChSystemSMC sys;sys.SetGravitationalAcceleration(chrono::ChVector3d(0,0,-9.81));
            auto f=fixture(sys,m,.3803912033738473,.04778053366332458);
            double xp=0,vp=0,ap=0,locked=0;
            for(int i=0;i<=int(std::round(.12/dt));++i){
                if(i>0)sys.DoStepDynamics(dt);
                double t=sys.GetChTime();auto s=m.at(t);auto p=f.tank->GetPos(),v=f.tank->GetPosDt(),a=f.tank->GetPosDt2();
                double pe=std::abs(p.x()-s.x),ve=std::abs(v.x()-s.v),ae=std::abs(a.x()-s.a);
                double lock=std::abs(p.y())+std::abs(p.z()-.04778053366332458)+f.tank->GetAngVelParent().Length()+std::abs(f.tank->GetRot().e0()-1);
                xp=std::max(xp,pe);vp=std::max(vp,ve);ap=std::max(ap,ae);locked=std::max(locked,lock);
                if(log)log<<m.scenario<<","<<dt<<","<<t<<","<<s.x<<","<<p.x()<<","<<s.v<<","<<v.x()<<","<<s.a<<","<<a.x()<<","<<pe<<","<<ve<<","<<ae<<","<<lock<<"\n";
            }
            std::cout<<name<<" dt="<<dt<<" position="<<xp<<" velocity="<<vp<<" acceleration="<<ap<<" locked="<<locked<<"\n";
            check(xp<1e-8&&vp<2e-5&&ap<.003&&locked<1e-10,"no fluid motor tracking");
        }
    }
    std::cout<<"checks="<<checks<<" pass="<<pass<<"\n";return pass?0:1;
}
