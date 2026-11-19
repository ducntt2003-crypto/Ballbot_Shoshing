"""Independent finite convex active-set oracle; exact binary64 input rationals.

Each triangle: interior plane minimizer (if feasible) and three bounded segment
minimizers; global min over this finite set is exact on the triangle union.
This does not reuse production Ericson regions, TIE, MERGE or union tolerances.
"""
from fractions import Fraction as F
from decimal import Decimal,localcontext
import numpy as np

def vec(v):return tuple(F(float(x)) for x in v)
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def mul(a,t):return tuple(x*t for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def triangle(p,tri):
 a,b,c=map(vec,tri);u,v,w=sub(b,a),sub(c,a),sub(p,a);uu,uv,vv=dot(u,u),dot(u,v),dot(v,v);det=uu*vv-uv*uv
 if det<=0:raise ValueError('Degenerate triangle')
 s=(dot(w,u)*vv-dot(w,v)*uv)/det;t=(dot(w,v)*uu-dot(w,u)*uv)/det;options=[]
 if s>=0 and t>=0 and s+t<=1:options.append((add(a,add(mul(u,s),mul(v,t))),'FACE',(s,t)))
 for x,y in [(a,b),(b,c),(c,a)]:
  d=sub(y,x);h=max(F(0),min(F(1),dot(sub(p,x),d)/dot(d,d)));options.append((add(x,mul(d,h)),'SEGMENT',None))
 q,feature,bary=min(options,key=lambda o:dot(sub(p,o[0]),sub(p,o[0])));d2=dot(sub(p,q),sub(p,q));kkt=tuple(dot(sub(p,q),sub(x,q)) for x in [a,b,c]);assert max(kkt)<=0
 return dict(q=q,d2=d2,feature=feature,bary=bary,kkt=kkt,det=det)
def solve(points,p):
 p=vec(p);results=[triangle(p,t) for t in points];best=min(r['d2'] for r in results);winning=[i for i,r in enumerate(results) if r['d2']==best];unique={results[i]['q'] for i in winning};index=winning[0];return dict(unique=len(unique)==1,global_points=len(unique),winning_triangles=winning,index=index,q=results[index]['q'],d2=best,results=results)
def decimal_sqrt(f):
 with localcontext() as c:c.prec=70;return (Decimal(f.numerator)/Decimal(f.denominator)).sqrt()
def serial(r):
 return dict(closest=[float(x) for x in r['q']],squared_distance_exact=str(r['d2']),distance_decimal=str(decimal_sqrt(r['d2'])),feature=r['feature'],barycentric=[float(x) for x in r['bary']] if r['bary'] else None,KKT_vertex_inner_products=[float(x) for x in r['kkt']],Gram_determinant_positive=r['det']>0)
