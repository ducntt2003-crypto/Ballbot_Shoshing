#pragma once
// NON_VALIDATED_EXPLORATORY. Analytic SI motion; no fluid equations here.
#include <cmath>
#include <stdexcept>
#include <string>
#include "chrono/functions/ChFunctionBase.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChLinkMotorLinearPosition.h"
namespace moving_tank {
struct Sample { double x=0,v=0,a=0; };
struct Motion {
    std::string scenario="STATIC";
    double acceleration=.1, ramp=.02, hold=.02, stop=.04;
    double amplitude=.001, frequency=1, envelope=1;
    static double s(double u){return u*u*u*(10+u*(-15+6*u));}
    static double ds(double u){return 30*u*u*(1-u)*(1-u);}
    static double dds(double u){return 60*u*(1-u)*(1-2*u);}
    static double S(double u){return std::pow(u,4)*(2.5-3*u+u*u);}
    static double J(double u){return std::pow(u,5)*(.5-.5*u+u*u/7);}
    Sample at(double t) const {
        if(t<0) return {};
        if(scenario=="STATIC")return {};
        if(scenario=="SINUSOIDAL"){
            double w=2*3.14159265358979323846*frequency,r=1,rd=0,rdd=0;
            if(envelope>0 && t<envelope){double u=t/envelope;r=s(u);rd=ds(u)/envelope;rdd=dds(u)/(envelope*envelope);}
            return {amplitude*r*std::sin(w*t),amplitude*(rd*std::sin(w*t)+r*w*std::cos(w*t)),
                amplitude*(rdd*std::sin(w*t)+2*rd*w*std::cos(w*t)-r*w*w*std::sin(w*t))};
        }
        if(scenario!="ACCELERATION" && scenario!="ACCELERATE_STOP")throw std::runtime_error("Unknown scenario");
        if(t<ramp){double u=t/ramp;return {acceleration*ramp*ramp*J(u),acceleration*ramp*S(u),acceleration*s(u)};}
        Sample b{acceleration*ramp*ramp/7,acceleration*ramp/2,acceleration};
        double z=t-ramp;
        if(z<hold)return {b.x+b.v*z+.5*acceleration*z*z,b.v+acceleration*z,acceleration};
        b={b.x+b.v*hold+.5*acceleration*hold*hold,b.v+acceleration*hold,acceleration};z-=hold;
        if(z<ramp){double u=z/ramp;return {b.x+b.v*z+acceleration*ramp*ramp*(.5*u*u-J(u)),b.v+acceleration*ramp*(u-S(u)),acceleration*(1-s(u))};}
        b={b.x+b.v*ramp+acceleration*ramp*ramp*(.5-1.0/7),b.v+acceleration*ramp/2,0};z-=ramp;
        if(scenario=="ACCELERATION")return {b.x+b.v*z,b.v,0};
        if(z<stop){double u=z/stop;return {b.x+b.v*stop*(u-S(u)),b.v*(1-s(u)),-b.v/stop*ds(u)};}
        return {b.x+b.v*stop/2,0,0};
    }
};
class MotionFunction : public chrono::ChFunction {
  public:
    explicit MotionFunction(Motion m):motion(m){}
    MotionFunction* Clone() const override {return new MotionFunction(*this);}
    double GetVal(double t) const override {return motion.at(t).x;}
    double GetDer(double t) const override {return motion.at(t).v;}
    double GetDer2(double t) const override {return motion.at(t).a;}
  private: Motion motion;
};
struct Fixture {
    std::shared_ptr<chrono::ChBody> tank,ground;
    std::shared_ptr<chrono::ChLinkMotorLinearPosition> motor;
};
inline Fixture fixture(chrono::ChSystemSMC& mbs,const Motion& motion,double mass,double cz,double initial_time=0){
    using namespace chrono;
    Fixture f;
    f.ground=chrono_types::make_shared<ChBody>();f.ground->SetFixed(true);mbs.AddBody(f.ground);
    f.tank=chrono_types::make_shared<ChBody>();f.tank->SetName("NON_VALIDATED_prescribed_tank");
    f.tank->SetFixed(false);f.tank->EnableCollision(false);f.tank->SetMass(mass);
    // Fixture-only positive inertia, NOT a calibrated physical tank inertia. All rotations constrained.
    f.tank->SetInertiaXX(ChVector3d(.001,.001,.001));
    f.tank->SetPos(ChVector3d(0,0,cz));f.tank->SetRot(QUNIT);mbs.AddBody(f.tank);
    f.motor=chrono_types::make_shared<ChLinkMotorLinearPosition>();
    f.motor->SetGuideConstraint(ChLinkMotorLinear::GuideConstraint::PRISMATIC);
    f.motor->Initialize(f.tank,f.ground,ChFramed(ChVector3d(0,0,cz),Q_ROTATE_Z_TO_X));
    f.motor->SetMotorFunction(chrono_types::make_shared<MotionFunction>(motion));mbs.AddLink(f.motor);
    // Project positions to rheonomic constraints; fluid remains RK2 unchanged.
    mbs.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT_PROJECTED);
    auto s=motion.at(initial_time);mbs.SetChTime(initial_time);
    f.tank->SetPos(ChVector3d(s.x,0,cz));f.tank->SetPosDt(ChVector3d(s.v,0,0));f.tank->SetPosDt2(ChVector3d(s.a,0,0));
    return f;
}
}
