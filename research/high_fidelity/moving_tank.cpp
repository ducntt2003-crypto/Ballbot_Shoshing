// NON_VALIDATED_EXPLORATORY prescribed tank. Baseline SPH/BCE setup copied from ppst_tank.cpp.
// No modifications to Chrono equations. All fluid diagnostics exclude BCE markers.
#include "Motion.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include "chrono/physics/ChSystemSMC.h"
#include "chrono_fsi/sph/ChFsiSystemSPH.h"
#include "chrono_thirdparty/rapidjson/document.h"
#include "chrono_thirdparty/rapidjson/istreamwrapper.h"

using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::fsi::sph;
namespace fs = std::filesystem;
static volatile std::sig_atomic_t interrupted = 0;
static void signal_stop(int) { interrupted = 1; }
static constexpr double PI = 3.14159265358979323846;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
struct Config {
    rapidjson::Document doc;
    explicit Config(const fs::path& path) {
        std::ifstream in(path); if(!in) throw std::runtime_error("Cannot read config");
        rapidjson::IStreamWrapper stream(in);doc.ParseStream(stream);
        if(doc.HasParseError() || !doc.IsObject()) throw std::runtime_error("Invalid JSON config");
    }
    double d(const char* k) const {if(!doc.HasMember(k)||!doc[k].IsNumber())throw std::runtime_error(k);return doc[k].GetDouble();}
    int i(const char* k) const {if(!doc.HasMember(k)||!doc[k].IsInt())throw std::runtime_error(k);return doc[k].GetInt();}
    bool b(const char* k) const {if(!doc.HasMember(k)||!doc[k].IsBool())throw std::runtime_error(k);return doc[k].GetBool();}
    std::string s(const char* k) const {if(!doc.HasMember(k)||!doc[k].IsString())throw std::runtime_error(k);return doc[k].GetString();}
};
struct Reference {
    double rho0,g,c,H;
    double p(double z) const {
        double d=std::max(0.0,H-z);
        return rho0*c*c/7 * (std::pow(1+6*g*d/(c*c),7.0/6)-1);
    }
    double rho(double z) const {return rho0*std::pow(1+6*g*std::max(0.0,H-z)/(c*c),1.0/6);}
};

int main(int argc,char** argv) try {
    if(argc!=3 && argc!=4)throw std::runtime_error("Usage: moving_tank config.json NEW_run_dir [--interface-only]");
    bool interface_only=argc==4 && std::string(argv[3])=="--interface-only";
    if(argc==4 && !interface_only)throw std::runtime_error("Unknown option");
    auto start=Clock::now();Config c(argv[1]);fs::path out=fs::absolute(argv[2]);
    if(!fs::create_directory(out))throw std::runtime_error("Refuse existing output directory");
    fs::create_directory(out/"particles");fs::create_directory(out/"bce");fs::copy_file(argv[1],out/"config.json");
    std::signal(SIGTERM,signal_stop);std::signal(SIGINT,signal_stop);
    moving_tank::Motion motion;motion.scenario=c.s("scenario");
    motion.acceleration=c.d("motion_acceleration_m_s2");motion.ramp=c.d("motion_ramp_s");
    motion.hold=c.d("motion_hold_s");motion.stop=c.d("motion_stop_s");
    motion.amplitude=c.d("sine_amplitude_m");motion.frequency=c.d("sine_frequency_Hz");motion.envelope=c.d("sine_envelope_s");
    double initial_time=c.d("initial_pose_time_s");
    if(!interface_only && initial_time!=0)throw std::runtime_error("Fluid runs must start at t=0");
    if(!(motion.ramp>0&&motion.hold>=0&&motion.stop>0&&motion.frequency>0&&motion.envelope>=0&&motion.amplitude>=0))throw std::runtime_error("Invalid motion parameters");
    double R=c.d("inner_radius_m"),H=c.d("inner_height_m"),dx=c.d("spacing_m");
    double rho0=c.d("rho0_kg_m3"),g=c.d("gravity_m_s2"),speed=c.d("sound_speed_m_s");
    double fill=c.d("nominal_water_volume_m3")/(PI*R*R),dt=c.d("timestep_s");
    int steps=c.i("steps"),stride=c.i("snapshot_every_steps");
    if(interface_only)steps=0;
    if(!(R>0&&H>fill&&fill>dx&&dt>0&&steps>=0&&steps*dt<=.020000001&&stride>0&&c.d("wall_clock_limit_s")<=115))throw std::runtime_error("Config geometry/time invalid");
    double h=c.d("h_over_spacing")*dx;
    if(dt*speed/h>.25)throw std::runtime_error("Initial acoustic CFL exceeds configured design bound .25");
    Reference ref{rho0,g,speed,fill};
    ChSystemSMC mbs;
    ChFsiFluidSystemSPH fluid;
    ChFsiSystemSPH fsi(&mbs,&fluid);
    fsi.SetVerbose(false);fsi.SetGravitationalAcceleration(ChVector3d(0,0,-g));
    fluid.SetVerbose(true); // engine PrintParams confirms actual installed settings
    fsi.SetStepSizeCFD(dt);fsi.SetStepsizeMBD(dt);
    ChFsiFluidSystemSPH::FluidProperties props;
    props.density=rho0;props.viscosity=c.d("dynamic_viscosity_Pa_s");props.char_length=2*R;
    fluid.SetCfdSPH(props);
    ChFsiFluidSystemSPH::SPHParameters sp;
    sp.integration_scheme=IntegrationScheme::RK2;sp.eos_type=EosType::TAIT;
    sp.viscosity_method=ViscosityMethod::LAMINAR;sp.boundary_method=BoundaryMethod::ADAMI;
    sp.kernel_type=KernelType::CUBIC_SPLINE;
    std::string shift=c.s("shifting_method");
    if(shift=="NONE")sp.shifting_method=ShiftingMethod::NONE;
    
    else throw std::runtime_error("Moving prototype requires baseline NONE");
    sp.shifting_ppst_push=c.d("ppst_push");sp.shifting_ppst_pull=c.d("ppst_pull");
    sp.initial_spacing=dx;sp.d0_multiplier=c.d("h_over_spacing");sp.max_velocity=speed/10;
    sp.num_bce_layers=c.i("bce_layers");sp.num_proximity_search_steps=1;
    sp.use_consistent_gradient_discretization=false;sp.use_consistent_laplacian_discretization=false;
    sp.use_delta_sph=c.b("delta_sph");sp.delta_sph_coefficient=c.d("delta_sph_coefficient");
    sp.artificial_viscosity=0;sp.density_reinit_steps=200000000;sp.use_variable_time_step=false;
    fluid.SetSPHParameters(sp);
    // Fluid centers on a Cartesian lattice. No renormalization to force nominal mass.
    int nr=static_cast<int>(std::ceil(R/dx));int nz=static_cast<int>(std::floor(fill/dx));
    std::vector<ChVector3d> initial;
    for(int k=0;k<nz;++k)for(int i=-nr;i<=nr;++i)for(int j=-nr;j<=nr;++j) {
        double x=i*dx,y=j*dx,z=(k+.5)*dx;
        if(x*x+y*y<R*R) {
            initial.emplace_back(x,y,z);
            fluid.AddSPHParticle(initial.back()+ChVector3d(motion.at(initial_time).x,0,0),ref.rho(z),ref.p(z),props.viscosity);
        }
    }
    // Official exterior-cylinder helper adds both caps; remove top cap for open tank.
    std::vector<ChVector3d> bce;size_t bottom_count=0,side_count=0;
    double tolerance=1e-10;
    if(c.s("bce_layout")=="native_polar") {
        auto closed=fluid.CreatePointsCylinderExterior(R,H,true);
        for(auto p:closed) {
            p.z()+=H/2;double r=std::hypot(p.x(),p.y());
            if(p.z()>H+tolerance)continue;
            if(r<R-tolerance && p.z()>tolerance)continue;
            if(r<R-tolerance){++bottom_count;}else{++side_count;}
            bce.push_back(p);
        }
    } else if(c.s("bce_layout")=="cartesian_complement_v1") {
        // Complement of the SAME fluid lattice. No force/EOS/mass correction.
        // Physical surfaces unchanged: cylinder r=R, floor z=0, open top z=H.
        int L=c.i("bce_layers"),N=static_cast<int>(std::ceil((R+L*dx)/dx));
        int Nz=static_cast<int>(std::ceil(H/dx));
        for(int k=-L;k<Nz;++k)for(int i=-N;i<=N;++i)for(int j=-N;j<=N;++j) {
            double x=i*dx,y=j*dx,z=(k+.5)*dx,r=std::hypot(x,y);
            if(z>=H || r>R+L*dx || !(z<0 || r>=R))continue;
            bce.emplace_back(x,y,z);
            if(r<R-tolerance)++bottom_count;else ++side_count;
        }
    } else {
        throw std::runtime_error("Unsupported BCE layout");
    }
    if(!bottom_count||!side_count||initial.empty())throw std::runtime_error("Empty boundary/fluid");
    // Tank COM C and floor T are different points; markers are defined in T.
    double cz=c.d("tank_com_above_floor_m");
    auto fixture=moving_tank::fixture(mbs,motion,c.d("tank_dry_mass_kg"),cz,initial_time);
    auto tank=fixture.tank;
    fsi.AddFsiBody(tank,bce,ChFramed(ChVector3d(0,0,-cz),QUNIT),false);
    double buffer=std::max(.05,6*h);
    fluid.SetComputationalDomain(ChAABB(ChVector3d(-R-buffer,-R-buffer,-buffer),ChVector3d(R+buffer,R+buffer,H+buffer)),BC_NONE);
    std::ofstream markers(out/"bce_markers.csv");markers<<std::setprecision(17)<<"x_m,y_m,z_m,nx_inward,ny_inward,nz_inward\n";
    for(auto p:bce){double r=std::hypot(p.x(),p.y());ChVector3d n(0,0,0);
        if(r>=R-tolerance)n=ChVector3d(-p.x()/r,-p.y()/r,0);
        if(p.z()<=tolerance)n+=ChVector3d(0,0,1);
        if(n.Length()>0)n.Normalize();
        markers<<p.x()<<","<<p.y()<<","<<p.z()<<","<<n.x()<<","<<n.y()<<","<<n.z()<<"\n";
    }markers.close();
    fsi.Initialize();fsi.SynchronizeDevice();
    size_t n=fluid.GetNumFluidMarkers();double pm=fluid.GetParticleMass(),mass=n*pm;
    if(n!=initial.size() || fluid.GetNumRigidBodyMarkers()!=bce.size())throw std::runtime_error("Marker count mismatch");
    std::ofstream meta(out/"run_meta.json");meta<<std::setprecision(17);
    meta<<"{\"label\":\"NON_VALIDATED_EXPLORATORY\",\"chrono_version\":\"10.0.0\",\"commit\":\"9faf13dd8f1128dd75ed233a9627027b0422c3f7\",\"scenario\":\""<<motion.scenario<<"\",\"interface_only\":"<<(interface_only?"true":"false")<<",\"fluid_particles\":"<<n<<",\"rigid_bce_markers\":"<<bce.size()<<",\"particle_mass_kg\":"<<pm<<",\"actual_mass_kg\":"<<mass<<",\"dt_s\":"<<dt<<",\"nominal_height_m\":"<<fill<<",\"smoothing_length_m\":"<<h<<",\"sound_speed_m_s\":"<<fluid.GetSoundSpeed()<<",\"torque_reference\":\"C=dry COM, W axes; tauT shifted to moving floor T\",\"fixture_inertia_kg_m2\":[0.001,0.001,0.001],\"inertia_calibrated\":false}\n";meta.close();
    std::ofstream diag(out/"diagnostics.csv");diag<<std::setprecision(17);
    diag<<"step,time_s,mbs_time_s,fluid_time_s,fsi_time_s,tank_x_m,tank_y_m,tank_z_floor_m,tank_vx_m_s,tank_ax_m_s2,command_x_m,command_v_m_s,command_a_m_s2,position_error_m,velocity_error_m_s,acceleration_error_m_s2,locked_dof_error,BCE_position_error_m,BCE_velocity_error_m_s,pressure_rmse_Pa,pressure_min_Pa,pressure_max_Pa,density_mean_kg_m3,density_min_kg_m3,density_max_kg_m3,density_relative_rmse,com_Wx_m,com_Wy_m,com_Wz_m,com_Tx_m,com_Ty_m,com_Tz_m,rms_world_velocity_m_s,max_world_velocity_m_s,rms_relative_velocity_m_s,max_relative_velocity_m_s,K_world_J,K_relative_J,mass_kg,ideal_outside,escaped,radial_penetration_m,bottom_penetration_m,top_penetration_m,force_valid,Fx_W_N,Fy_W_N,Fz_W_N,tauCx_W_Nm,tauCy_W_Nm,tauCz_W_Nm,tauTx_W_Nm,tauTy_W_Nm,tauTz_W_Nm,actuator_force_N,fluid_fd_max_acceleration_m_s2,momentum_balance_defect_N,boundary_power_proxy_W,cumulative_boundary_work_proxy_J,physics_wall_s,wall_s\n";
    std::ofstream status_file;std::string status="complete";int completed=0;
    ChVector3d prevP(0,0,0);std::vector<ChVector3d> prevV;
    double physics_total=0,physics_max=0,power_integral=0,previous_power=0;
    double max_bp=0,max_bv=0,max_pe=0,max_ve=0,max_ae=0;
    for(int step=0;step<=steps;++step){
        double physics=0;
        if(step>0){
            if(interrupted||seconds(start)>c.d("wall_clock_limit_s")){status="wall_clock_stop";break;}
            auto tick=Clock::now();fsi.DoStepDynamics(dt);fsi.SynchronizeDevice();physics=seconds(tick);
            physics_total+=physics;physics_max=std::max(physics_max,physics);
        }
        double time=initial_time+step*dt;auto cmd=motion.at(time);
        auto pp=fluid.GetParticlePositions(),vv=fluid.GetParticleVelocities(),qq=fluid.GetParticleFluidProperties();
        if(pp.size()!=n+bce.size() || vv.size()!=pp.size() || qq.size()!=pp.size() || fluid.GetNumBoundaryMarkers()!=0)throw std::runtime_error("Unexpected original-order marker layout");
        auto bodypos=tank->GetPos(),bodyvel=tank->GetPosDt(),bodyacc=tank->GetPosDt2();
        auto floor=tank->TransformPointLocalToParent(ChVector3d(0,0,-cz));
        auto omega=tank->GetAngVelParent();double bp=0,bv=0;
        for(size_t j=0;j<bce.size();++j){auto local=bce[j]-ChVector3d(0,0,cz);
            auto expected=tank->TransformPointLocalToParent(local);
            auto ev=bodyvel+Vcross(omega,expected-bodypos);
            bp=std::max(bp,(pp[n+j]-expected).Length());bv=std::max(bv,(vv[n+j]-ev).Length());
        }
        double pe=(bodypos-ChVector3d(cmd.x,0,cz)).Length(),ve=(bodyvel-ChVector3d(cmd.v,0,0)).Length(),ae=(bodyacc-ChVector3d(cmd.a,0,0)).Length();
        double locked=std::abs(bodypos.y())+std::abs(bodypos.z()-cz)+omega.Length()+std::abs(tank->GetRot().e0()-1);
        max_bp=std::max(max_bp,bp);max_bv=std::max(max_bv,bv);max_pe=std::max(max_pe,pe);max_ve=std::max(max_ve,ve);max_ae=std::max(max_ae,ae);
        double vsq=0,relsq=0,vmax=0,relmax=0,pmin=1e100,pmax=-1e100,rmin=1e100,rmax=-1e100,rmean=0,rsq=0,pse=0,fdamax=0,pr=0,pb=0,pt=0;
        int outside=0,escaped=0;bool finite=true;ChVector3d com(0,0,0),P(0,0,0);
        for(size_t i=0;i<n;++i){auto p=pp[i],v=vv[i];auto rel=tank->TransformDirectionParentToLocal(v-bodyvel-Vcross(omega,p-bodypos));
            auto pT=tank->TransformPointParentToLocal(p)+ChVector3d(0,0,cz);
            double rho=qq[i].x(),pressure=qq[i].y();
            finite=finite&&std::isfinite(p.x()+p.y()+p.z()+v.Length2()+rho+pressure)&&rho>0;
            vsq+=v.Length2();relsq+=rel.Length2();vmax=std::max(vmax,v.Length());relmax=std::max(relmax,rel.Length());
            if(step>0)fdamax=std::max(fdamax,((v-prevV[i])/dt).Length());
            pmin=std::min(pmin,pressure);pmax=std::max(pmax,pressure);rmin=std::min(rmin,rho);rmax=std::max(rmax,rho);rmean+=rho;rsq+=std::pow(rho/rho0-1,2);
            pse+=std::pow(pressure-rho0*g*std::max(0.0,fill-pT.z()),2);com+=p;P+=pm*v;
            double dr=std::hypot(pT.x(),pT.y())-R,db=-pT.z(),dh=pT.z()-H;
            pr=std::max(pr,dr);pb=std::max(pb,db);pt=std::max(pt,dh);
            if(std::max({dr,db,dh})>0)++outside;
            if(std::max({dr,db,dh})>dx/2)++escaped;
        }
        com/=static_cast<double>(n);auto comT=tank->TransformPointParentToLocal(com)+ChVector3d(0,0,cz);
        auto f=fsi.GetFsiBodyForce(0),tc=fsi.GetFsiBodyTorque(0),tt=tc+Vcross(bodypos-floor,f);
        finite=finite&&std::isfinite(f.Length2()+tc.Length2()+bodypos.Length2()+bodyvel.Length2()+bodyacc.Length2());
        double defect=step>0?((P-prevP)/dt-(ChVector3d(0,0,-mass*g)-f)).Length():0;
        // End-state rectangular/trapezoid proxy, not native RK stage-resolved work.
        double power=-Vdot(f,bodyvel)-Vdot(tc,omega);
        if(step>0)power_integral+=.5*(power+previous_power)*dt;
        previous_power=power;prevP=P;prevV.assign(vv.begin(),vv.begin()+n);
        double clock_error=interface_only?0:std::max({std::abs(mbs.GetChTime()-time),std::abs(fluid.GetSimTime()-time),std::abs(fsi.GetSimTime()-time)});
        if(!finite)status="nonfinite_stop";
        else if(pe>1e-8||ve>2e-5||locked>1e-10||bp>1e-10||bv>1e-10||clock_error>1e-10)status="motion_interface_stop";
        else if(escaped||std::max(std::abs(rmax/rho0-1),std::abs(rmin/rho0-1))>c.d("safety_max_relative_density_error")||std::max(std::abs(pmin),std::abs(pmax))>c.d("safety_pressure_abs_Pa")||relmax>c.d("safety_relative_speed_m_s"))status="fluid_safety_stop";
        diag<<step<<","<<time<<","<<mbs.GetChTime()<<","<<fluid.GetSimTime()<<","<<fsi.GetSimTime()<<","<<floor.x()<<","<<floor.y()<<","<<floor.z()<<","<<bodyvel.x()<<","<<bodyacc.x()<<","<<cmd.x<<","<<cmd.v<<","<<cmd.a<<","<<pe<<","<<ve<<","<<ae<<","<<locked<<","<<bp<<","<<bv<<","<<std::sqrt(pse/n)<<","<<pmin<<","<<pmax<<","<<rmean/n<<","<<rmin<<","<<rmax<<","<<std::sqrt(rsq/n)<<","<<com.x()<<","<<com.y()<<","<<com.z()<<","<<comT.x()<<","<<comT.y()<<","<<comT.z()<<","<<std::sqrt(vsq/n)<<","<<vmax<<","<<std::sqrt(relsq/n)<<","<<relmax<<","<<.5*pm*vsq<<","<<.5*pm*relsq<<","<<mass<<","<<outside<<","<<escaped<<","<<pr<<","<<pb<<","<<pt<<","<<(step>0)<<","<<f.x()<<","<<f.y()<<","<<f.z()<<","<<tc.x()<<","<<tc.y()<<","<<tc.z()<<","<<tt.x()<<","<<tt.y()<<","<<tt.z()<<","<<fixture.motor->GetMotorForce()<<","<<fdamax<<","<<defect<<","<<power<<","<<power_integral<<","<<physics<<","<<seconds(start)<<"\n";diag.flush();
        if(step%stride==0 || step==steps || status!="complete"){
            std::ostringstream name;name<<"frame_"<<std::setw(6)<<std::setfill('0')<<step<<".csv";
            std::ofstream frame(out/"particles"/name.str());frame<<std::setprecision(17)<<"id,time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,pressure_Pa,density_kg_m3\n";
            for(size_t i=0;i<n;++i)frame<<i<<","<<time<<","<<pp[i].x()<<","<<pp[i].y()<<","<<pp[i].z()<<","<<vv[i].x()<<","<<vv[i].y()<<","<<vv[i].z()<<","<<qq[i].y()<<","<<qq[i].x()<<"\n";
            std::ofstream bm(out/"bce"/name.str());bm<<std::setprecision(17)<<"id,time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s\n";
            for(size_t j=0;j<bce.size();++j)bm<<j<<","<<time<<","<<pp[n+j].x()<<","<<pp[n+j].y()<<","<<pp[n+j].z()<<","<<vv[n+j].x()<<","<<vv[n+j].y()<<","<<vv[n+j].z()<<"\n";
        }
        completed=step;if(status!="complete")break;
    }
    std::ofstream final(out/"run_status.json");final<<std::setprecision(17)<<"{\"label\":\"NON_VALIDATED_EXPLORATORY\",\"status\":\""<<status<<"\",\"completed_steps\":"<<completed<<",\"simulated_seconds\":"<<completed*dt<<",\"physics_seconds\":"<<physics_total<<",\"max_physics_step_seconds\":"<<physics_max<<",\"wall_seconds\":"<<seconds(start)<<",\"BCE_position_error_max_m\":"<<max_bp<<",\"BCE_velocity_error_max_m_s\":"<<max_bv<<",\"tank_position_error_max_m\":"<<max_pe<<",\"tank_velocity_error_max_m_s\":"<<max_ve<<",\"tank_acceleration_error_max_m_s2\":"<<max_ae<<"}\n";
    std::cout<<"NON_VALIDATED_EXPLORATORY "<<motion.scenario<<" "<<status<<" steps="<<completed<<"\n";
    return status=="complete"?0:2;
} catch(const std::exception& e){std::cerr<<"MOVING_TANK_ERROR "<<e.what()<<"\n";return 1;}
