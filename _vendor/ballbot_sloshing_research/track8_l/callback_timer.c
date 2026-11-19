#define _POSIX_C_SOURCE 200809L
#include <mujoco/mujoco.h>
#include <time.h>
static mjfCollision saved=0;static double times[2000000];static int n=0,overflow=0;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int timed(const mjModel*m,const mjData*d,mjContact*c,int a,int b,mjtNum margin){double t=now();int r=saved(m,d,c,a,b,margin);double v=now()-t;if(n<2000000)times[n++]=v;else overflow++;return r;}
int timer_install(void){if(saved)return 0;saved=mjCOLLISIONFUNC[mjGEOM_HFIELD][mjGEOM_SPHERE];n=overflow=0;mjCOLLISIONFUNC[mjGEOM_HFIELD][mjGEOM_SPHERE]=timed;return 1;}
int timer_uninstall(void){if(!saved)return 0;mjCOLLISIONFUNC[mjGEOM_HFIELD][mjGEOM_SPHERE]=saved;saved=0;return 1;}
int timer_read(double*out){for(int i=0;i<n;i++)out[i]=times[i];return n;}
int timer_count(void){return n;}
int timer_overflow(void){return overflow;}
