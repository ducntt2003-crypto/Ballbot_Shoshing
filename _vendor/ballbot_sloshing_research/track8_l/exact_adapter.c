/* TRACK8-G: frozenFdistance/patchgraph plus UNION vertex-KKT filter and abort. */
/* One unique exposed closest point per connected overlap patch.
 * Uses unchanged MuJoCo soft point-contact law; NOT a distributed elastic patch.
 * Union-stationarity filtering replaces F's near-distance-only classification.
 * Unsupported physical stages raise the public API error, NEVER nativefallback.
 * Public process-local collision dispatch, no engine/core/binary modification. */
#include <mujoco/mujoco.h>
#include <math.h>
#include <string.h>
#include <float.h>
#define MAXTRI 8192
#define TIE 1e-12
#define MERGE2 1e-16
typedef int (*ExactSelector)(int,const double*,const double*,double*);
static ExactSelector exact_selector=0;
static unsigned long exact_calls=0,exact_resolved=0,exact_rejected=0;
void track8l_set_selector(ExactSelector f){exact_selector=f;}
void track8l_exact_counts(unsigned long*out){out[0]=exact_calls;out[1]=exact_resolved;out[2]=exact_rejected;}
static mjfCollision saved=0;
static unsigned long calls=0,accepted=0,fallbacks[8]={0},nt=0,maxtri=0,maxactive=0;
static double times[64];
static int last_status=0,last_components=0;
static double last_point[3],last_normal[3],last_dist=0;
static int abort_mode=1,phase=0,step_evals=0,force_abort_at=0;
static int last_near=0,last_stationary=0,last_rejected=0;
static double last_violation=0;
static double event_state[512],event_time=0;
static int event_size=0,event_reason=0,event_phase=0,event_eval=0;

static double dot(const double*a,const double*b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static double normdiff2(const double*a,const double*b){double v[3];for(int k=0;k<3;k++)v[k]=a[k]-b[k];return dot(v,v);}
static void seg(double*q,const double*p,const double*a,const double*b){double ab[3],ap[3];for(int k=0;k<3;k++){ab[k]=b[k]-a[k];ap[k]=p[k]-a[k];}double t=dot(ap,ab)/dot(ab,ab);t=fmin(1,fmax(0,t));for(int k=0;k<3;k++)q[k]=a[k]+t*ab[k];}
/* Ericson region cases, independent from Python face/segment enumeration. */
static void closest(double*q,const double*p,const double*a,const double*b,const double*c){
 double ab[3],ac[3],ap[3],bp[3],cp[3];for(int k=0;k<3;k++){ab[k]=b[k]-a[k];ac[k]=c[k]-a[k];ap[k]=p[k]-a[k];bp[k]=p[k]-b[k];cp[k]=p[k]-c[k];}
 double d1=dot(ab,ap),d2=dot(ac,ap),d3=dot(ab,bp),d4=dot(ac,bp),d5=dot(ab,cp),d6=dot(ac,cp);
 if(d1<=0&&d2<=0){memcpy(q,a,3*sizeof(double));return;}
 if(d3>=0&&d4<=d3){memcpy(q,b,3*sizeof(double));return;}
 double vc=d1*d4-d3*d2;
 if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);for(int k=0;k<3;k++)q[k]=a[k]+v*ab[k];return;}
 if(d6>=0&&d5<=d6){memcpy(q,c,3*sizeof(double));return;}
 double vb=d5*d2-d1*d6;
 if(vb<=0&&d2>=0&&d6<=0){double w=d2/(d2-d6);for(int k=0;k<3;k++)q[k]=a[k]+w*ac[k];return;}
 double va=d3*d6-d5*d4;
 if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){double w=(d4-d3)/((d4-d3)+(d5-d6));for(int k=0;k<3;k++)q[k]=b[k]+w*(c[k]-b[k]);return;}
 double inv=1/(va+vb+vc),v=vb*inv,w=vc*inv;for(int k=0;k<3;k++)q[k]=a[k]+v*ab[k]+w*ac[k];
}
static void vertices(const mjModel*m,int hid,int row,int col,int type,double*v){
 int nc=m->hfield_ncol[hid];const double*s=m->hfield_size+4*hid;double dx=2*s[0]/(nc-1),dy=2*s[1]/(m->hfield_nrow[hid]-1);const float*h=m->hfield_data+m->hfield_adr[hid];
 int dr[2][3]={{0,0,1},{0,1,1}},dc[2][3]={{0,1,1},{0,1,0}};
 for(int k=0;k<3;k++){int r=row+dr[type][k],c=col+dc[type][k];v[3*k]=-s[0]+c*dx;v[3*k+1]=-s[1]+r*dy;v[3*k+2]=h[r*nc+c]*s[2];}
}

/* A local minimum on UNION must minimize squared distance on EACH incident
 * convex triangle. Its gradient must satisfy all feasible-vertex inequalities.
 * qerr uses conservative backward-error/Gram-conditioning scaling. */
static int stationary_union(const mjModel*m,int hid,const double*p,const double*q,double*outviol){
 const double*s=m->hfield_size+4*hid;int nc=m->hfield_ncol[hid],nr=m->hfield_nrow[hid];
 double dx=2*s[0]/(nc-1),dy=2*s[1]/(nr-1);
 int cc=(int)floor((q[0]+s[0])/dx),rr=(int)floor((q[1]+s[1])/dy),incident=0;
 for(int r=rr-1;r<=rr+1;r++)for(int c=cc-1;c<=cc+1;c++)if(r>=0&&r<nr-1&&c>=0&&c<nc-1)for(int type=0;type<2;type++){
  double v[9],ab[3],ac[3],w[3],norm[3];vertices(m,hid,r,c,type,v);double scale=1;
  for(int k=0;k<3;k++){ab[k]=v[3+k]-v[k];ac[k]=v[6+k]-v[k];w[k]=q[k]-v[k];scale=fmax(scale,1+fabs(p[k]));scale=fmax(scale,1+fabs(q[k]));}
  for(int k=0;k<9;k++)scale=fmax(scale,1+fabs(v[k]));
  double uu=dot(ab,ab),uv=dot(ab,ac),vv=dot(ac,ac),disc=hypot(uu-vv,2*uv),lmax=.5*(uu+vv+disc),lmin=.5*(uu+vv-disc);
  if(lmin<=64*DBL_EPSILON*lmax)return -1;
  double cond=lmax/lmin,err=128*DBL_EPSILON*cond*scale,det=uu*vv-uv*uv;
  norm[0]=ab[1]*ac[2]-ab[2]*ac[1];norm[1]=ab[2]*ac[0]-ab[0]*ac[2];norm[2]=ab[0]*ac[1]-ab[1]*ac[0];
  double plane=fabs(dot(w,norm))/sqrt(dot(norm,norm));double wu=dot(w,ab),wv=dot(w,ac),u=(wu*vv-wv*uv)/det,t=(wv*uu-wu*uv)/det;
  double tol=8*err/fmin(sqrt(uu),sqrt(vv));
  if(plane>8*err||u<-tol||t<-tol||u+t>1+tol)continue;
  incident++;
  double pq[3];for(int k=0;k<3;k++)pq[k]=p[k]-q[k];
  for(int j=0;j<3;j++){
   double vq[3],sumabs=0;for(int k=0;k<3;k++){vq[k]=v[3*j+k]-q[k];sumabs+=fabs(pq[k]*vq[k]);}
   double allowance=16*DBL_EPSILON*sumabs+8*err*(sqrt(dot(pq,pq))+sqrt(dot(vq,vq))+err);
   double violation=dot(pq,vq)-allowance;if(violation>*outviol)*outviol=violation;
   if(violation>0)return 0;
  }
 }
 return incident?1:-1;
}
/* Native unsupported domain NEVER falls back to a different contact law.
 * Python mj_step/mj_forward error interceptor converts mju_error to FatalError.
 * Query-mode return0 is only for non-integrating geometry fixtures. */
static int delegate(const mjModel*m,const mjData*d,mjContact*con,int g1,int g2,double margin,int why){
 (void)con;(void)g1;(void)g2;(void)margin;
 fallbacks[why]++;last_status=-why-1;event_time=d->time;event_reason=why;event_phase=phase;event_eval=step_evals;
 event_size=mj_stateSize(m,mjSTATE_INTEGRATION);
 if(event_size<=512)mj_getState(m,d,event_state,mjSTATE_INTEGRATION);
 if(abort_mode)mju_error("TRACK8G_UNSUPPORTED_%d time=%.17g phase=%d eval=%d",why,d->time,phase,step_evals);
 return 0;
}
static int exterior(const mjModel*m,const mjData*d,mjContact*con,int g1,int g2,mjtNum margin){
 calls++;if(nt<64)times[nt++]=d->time;last_status=0;last_components=0;
 last_near=last_stationary=last_rejected=0;last_violation=0;last_dist=NAN;
 for(int k=0;k<3;k++){last_point[k]=NAN;last_normal[k]=NAN;}
 if(phase==2){step_evals++;if(force_abort_at>0&&step_evals==force_abort_at)return delegate(m,d,con,g1,g2,margin,6);}
 int hid=m->geom_dataid[g1];if(hid<0||m->geom_type[g1]!=mjGEOM_HFIELD||m->geom_type[g2]!=mjGEOM_SPHERE)return delegate(m,d,con,g1,g2,margin,0);
 const double*s=m->hfield_size+4*hid;const double*R=d->geom_xmat+9*g1;double delta[3],p[3];mju_sub3(delta,d->geom_xpos+3*g2,d->geom_xpos+3*g1);mju_mulMatTVec(p,R,delta,3,3);double rad=m->geom_size[3*g2],bound=rad+margin;
 if(fabs(p[0])+bound>=s[0]||fabs(p[1])+bound>=s[1])return delegate(m,d,con,g1,g2,margin,1);
 int nc=m->hfield_ncol[hid],nr=m->hfield_nrow[hid];double dx=2*s[0]/(nc-1),dy=2*s[1]/(nr-1);
 int c0=(int)floor((p[0]-bound+s[0])/dx),c1=(int)floor((p[0]+bound+s[0])/dx),r0=(int)floor((p[1]-bound+s[1])/dy),r1=(int)floor((p[1]+bound+s[1])/dy);
 if(c0<0)c0=0;
 if(r0<0)r0=0;
 if(c1>nc-2)c1=nc-2;
 if(r1>nr-2)r1=nr-2;
 int cols=c1-c0+1,rows=r1-r0+1,count=2*cols*rows;if(count<=0||count>MAXTRI)return delegate(m,d,con,g1,g2,margin,2);
 if((unsigned long)count>maxtri)maxtri=count;
 double qs[MAXTRI][3],dist[MAXTRI],maximum=-1e100,minimum=1e100;unsigned char active[MAXTRI],seen[MAXTRI];int queue[MAXTRI],best=0,na=0;memset(seen,0,(size_t)count);
 for(int row=r0;row<=r1;row++)for(int col=c0;col<=c1;col++)for(int type=0;type<2;type++){
  int index=2*((row-r0)*cols+col-c0)+type;double v[9];vertices(m,hid,row,col,type,v);for(int j=0;j<3;j++)if(v[3*j+2]>maximum)maximum=v[3*j+2];closest(qs[index],p,v,v+3,v+6);dist[index]=sqrt(normdiff2(qs[index],p));active[index]=(dist[index]<=bound+1e-12);na+=active[index];if(dist[index]<minimum){minimum=dist[index];best=index;}
 }
 if((unsigned long)na>maxactive)maxactive=na;
 last_dist=minimum;
 for(int k=0;k<3;k++){last_point[k]=qs[best][k];last_normal[k]=(minimum>1e-12)?(p[k]-qs[best][k])/minimum:NAN;}
 if(p[2]<=maximum||minimum<1e-12)return delegate(m,d,con,g1,g2,margin,3);
 if(minimum>bound){last_status=0;return 0;}
 int components=0;
 for(int i=0;i<count;i++)if(active[i]&&!seen[i]){
  components++;int head=0,tail=0;queue[tail++]=i;seen[i]=1;
  while(head<tail){int index=queue[head++],type=index%2,cell=index/2,rr=cell/cols,cc=cell%cols;double v[9];vertices(m,hid,r0+rr,c0+cc,type,v);int nb[3],ea[3],eb[3];
   if(type==0){nb[0]=index+1;nb[1]=(cc+1<cols)?index+3:-1;nb[2]=(rr>0)?index-2*cols+1:-1;ea[0]=0;eb[0]=2;ea[1]=1;eb[1]=2;ea[2]=0;eb[2]=1;}
   else{nb[0]=index-1;nb[1]=(cc>0)?index-3:-1;nb[2]=(rr+1<rows)?index+2*cols-1:-1;ea[0]=0;eb[0]=1;ea[1]=0;eb[1]=2;ea[2]=1;eb[2]=2;}
   for(int k=0;k<3;k++)if(nb[k]>=0&&nb[k]<count&&active[nb[k]]&&!seen[nb[k]]){double q[3];seg(q,p,v+3*ea[k],v+3*eb[k]);if(normdiff2(q,p)<=bound*bound+1e-14){seen[nb[k]]=1;queue[tail++]=nb[k];}}
  }
 }
 last_components=components;if(components!=1)return delegate(m,d,con,g1,g2,margin,4);
 int selected=-1;unsigned char stationary_mask[MAXTRI];memset(stationary_mask,0,(size_t)count);
 for(int i=0;i<count;i++)if(dist[i]<=minimum+TIE){
  last_near++;double violation=0;int st=stationary_union(m,hid,p,qs[i],&violation);
  if(violation>last_violation)last_violation=violation;
  if(st<0)return delegate(m,d,con,g1,g2,margin,7);
  if(st){last_stationary++;stationary_mask[i]=1;if(selected<0||dist[i]<dist[selected])selected=i;}
  else last_rejected++;
 }
 if(selected<0)return delegate(m,d,con,g1,g2,margin,7);
 best=selected;minimum=dist[best];last_dist=minimum;
 for(int k=0;k<3;k++){last_point[k]=qs[best][k];last_normal[k]=(p[k]-qs[best][k])/minimum;}
 int ambiguous=0;
 for(int i=0;i<count;i++)if(stationary_mask[i]&&dist[i]<=minimum+TIE&&normdiff2(qs[i],qs[best])>MERGE2)ambiguous=1;
 if(ambiguous){
  exact_calls++;if(!exact_selector||count>512){exact_rejected++;return delegate(m,d,con,g1,g2,margin,5);}
  double mesh[MAXTRI][9],qexact[3];
  for(int row=r0;row<=r1;row++)for(int col=c0;col<=c1;col++)for(int type=0;type<2;type++){
   int index=2*((row-r0)*cols+col-c0)+type;vertices(m,hid,row,col,type,mesh[index]);
  }
  int chosen=exact_selector(count,p,&mesh[0][0],qexact);
  if(chosen<0||chosen>=count){exact_rejected++;return delegate(m,d,con,g1,g2,margin,5);}
  /* Exact minimizer rounded once; original point-contact law below unchanged. */
  best=chosen;for(int k=0;k<3;k++)qs[best][k]=qexact[k];
  minimum=sqrt(normdiff2(qs[best],p));last_dist=minimum;
  for(int k=0;k<3;k++){last_point[k]=qs[best][k];last_normal[k]=(p[k]-qs[best][k])/minimum;}
  exact_resolved++;
 }
 double normal[3],worldnormal[3];for(int k=0;k<3;k++)normal[k]=(p[k]-qs[best][k])/minimum;mju_mulMatVec3(worldnormal,R,normal);double gap=minimum-rad;con[0].dist=gap;mju_zero(con[0].frame,9);
 for(int k=0;k<3;k++){con[0].frame[k]=worldnormal[k];con[0].pos[k]=d->geom_xpos[3*g2+k]-(rad+.5*gap)*worldnormal[k];last_point[k]=qs[best][k];last_normal[k]=normal[k];}
 last_dist=minimum;last_status=1;accepted++;return 1;
}
int track8l_install(void){if(saved)return 0;saved=mjCOLLISIONFUNC[mjGEOM_HFIELD][mjGEOM_SPHERE];mjCOLLISIONFUNC[mjGEOM_HFIELD][mjGEOM_SPHERE]=exterior;return 1;}
int track8l_uninstall(void){if(!saved)return 0;mjCOLLISIONFUNC[mjGEOM_HFIELD][mjGEOM_SPHERE]=saved;saved=0;return 1;}
void track8l_reset(void){calls=accepted=nt=maxtri=maxactive=0;exact_calls=exact_resolved=exact_rejected=0;memset(fallbacks,0,sizeof(fallbacks));event_size=0;event_time=0;event_reason=0;event_phase=0;event_eval=0;}
unsigned long track8l_counts(unsigned long*out){out[0]=calls;out[1]=accepted;out[2]=maxtri;out[3]=maxactive;for(int i=0;i<8;i++)out[4+i]=fallbacks[i];return 12;}
unsigned long track8l_times(double*out){for(unsigned long i=0;i<nt;i++)out[i]=times[i];return nt;}
int track8l_last(double*out){out[0]=last_dist;for(int i=0;i<3;i++){out[1+i]=last_point[i];out[4+i]=last_normal[i];}out[7]=last_components;return last_status;}
int track8l_is_installed(void){return saved!=0;}

void track8l_mode(int abort_enabled){abort_mode=abort_enabled;}
int track8l_get_mode(void){return abort_mode;}
void track8l_phase(int p){phase=p;if(p==2)step_evals=0;}
void track8l_force(int eval){force_abort_at=eval;}
int track8l_classifier(double*out){out[0]=last_near;out[1]=last_stationary;out[2]=last_rejected;out[3]=last_violation;return last_status;}
int track8l_event(double*out,double*state){out[0]=event_time;out[1]=event_reason;out[2]=event_phase;out[3]=event_eval;out[4]=event_size;for(int i=0;i<event_size&&i<512;i++)state[i]=event_state[i];return event_size;}
