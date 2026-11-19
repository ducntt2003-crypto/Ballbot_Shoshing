/* Read-only PUBLIC control callback and delayed caches; no force modification. */
#include <mujoco/mujoco.h>
#include <string.h>
#define MAXE 8
#define MAXV 128
#define MAXS 512
static mjfGeneric oldcb=0;
static int installed=0,active=0,count=0,overflow=0,nv=0,ns=0,finished=0;
static const mjModel* model=0;static const mjData* data=0;
static double states[MAXE][MAXS],acc[MAXE][MAXV],constraint[MAXE][MAXV],times[MAXE];
static double finalacc[MAXV],finalconstraint[MAXV],directground[3];
static void record(const mjModel*m,mjData*d){
 if(oldcb)oldcb(m,d);
 if(!active)return;
 model=m;data=d;nv=m->nv;ns=mj_stateSize(m,mjSTATE_INTEGRATION);
 if(count>=MAXE||nv>MAXV||ns>MAXS){overflow=1;return;}
 times[count]=d->time;mj_getState(m,d,states[count],mjSTATE_INTEGRATION);
 memcpy(acc[count],d->qacc,nv*sizeof(double));memcpy(constraint[count],d->qfrc_constraint,nv*sizeof(double));count++;
}
int track8i_install(void){if(installed)return 0;oldcb=mjcb_control;mjcb_control=record;installed=1;return 1;}
int track8i_uninstall(void){if(!installed)return 0;mjcb_control=oldcb;oldcb=0;installed=active=0;return 1;}
int track8i_installed(void){return installed;}
void track8i_begin(void){count=overflow=finished=0;model=0;data=0;active=1;}
void track8i_cancel(void){active=0;}
int track8i_finish(void){
 active=0;if(!model||!data||nv>MAXV||ns>MAXS)return 0;
 memcpy(finalacc,data->qacc,nv*sizeof(double));memcpy(finalconstraint,data->qfrc_constraint,nv*sizeof(double));memset(directground,0,sizeof(directground));
 int ground=mj_name2id(model,mjOBJ_GEOM,"terrain");
 for(int ci=0;ci<data->ncon;ci++){
  const mjContact*c=data->contact+ci;if(c->geom[0]!=ground&&c->geom[1]!=ground)continue;
  double f[6];mj_contactForce(model,data,ci,f);int sign=c->geom[0]==ground?1:-1;
  for(int j=0;j<3;j++)for(int k=0;k<3;k++)directground[j]+=sign*c->frame[3*k+j]*f[k];
 }
 finished=1;return count;
}
void track8i_info(int*out){out[0]=count;out[1]=nv;out[2]=ns;out[3]=overflow;out[4]=finished;}
void track8i_read(double*st,double*a,double*f,double*t,double*last_a,double*last_f,double*w){
 for(int i=0;i<count;i++){memcpy(st+i*ns,states[i],ns*sizeof(double));memcpy(a+i*nv,acc[i],nv*sizeof(double));memcpy(f+i*nv,constraint[i],nv*sizeof(double));t[i]=times[i];}
 memcpy(last_a,finalacc,nv*sizeof(double));memcpy(last_f,finalconstraint,nv*sizeof(double));memcpy(w,directground,3*sizeof(double));
}
