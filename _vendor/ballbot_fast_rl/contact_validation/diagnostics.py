"""Independent analytic capsule/sphere gap, contact Jacobians and frame/force checks."""
import numpy as np
import mujoco

def solver_residuals(env):
    """Public solver diagnostics. Soft/cone residuals need not vanish componentwise."""
    m,d=env.model,env.data
    mass_acc=np.zeros(m.nv);mujoco.mj_mulM(m,d,mass_acc,d.qacc)
    eom=mass_acc-d.qfrc_smooth-d.qfrc_constraint
    jqacc=np.zeros(d.nefc);mujoco.mj_mulJacVec(m,d,jqacc,d.qacc)
    unprojected=jqacc-d.efc_aref+d.efc_R*d.efc_force
    return dict(equation_of_motion_max_abs_native_force_units=float(abs(eom).max()),soft_constraint_unprojected_balance_RMS=float(np.sqrt(np.mean(unprojected**2))) if d.nefc else 0.,scope='M*qacc - smooth - constraint in native generalized force units; unprojected J*qacc-aref+R*f is not a cone KKT violation norm; sampled endpoint, not RK4-stage proof')

def sphere_capsule(ball_center,ball_radius,center,axis,radius,half_length):
    axial=float(np.clip(np.dot(ball_center-center,axis),-half_length,half_length))
    closest=center+axial*axis
    return float(np.linalg.norm(ball_center-closest)-ball_radius-radius),closest,axial

def geometry(env):
    m,d=env.model,env.data;ball=d.geom_xpos[env.ball_geom].copy();radius=float(m.geom_size[env.ball_geom,0]);wheels=[]
    for k in range(3):
        gi=m.geom(f'wheel_mesh_{k}').id;ji=int(env.wheel_ids[k]);center=d.geom_xpos[gi].copy();axis=d.geom_xmat[gi].reshape(3,3)[:,2].copy();anchor=d.xanchor[ji].copy();ja=d.xaxis[ji].copy()
        gap,point,t=sphere_capsule(ball,radius,center,axis,float(m.geom_size[gi,0]),float(m.geom_size[gi,1]))
        wheels.append(dict(wheel=k,geom_name=m.geom(gi).name,geom_type=int(m.geom_type[gi]),radius_m=float(m.geom_size[gi,0]),capsule_half_segment_m=float(m.geom_size[gi,1]),capsule_total_axial_length_m=float(2*sum(m.geom_size[gi,:2])),center_W_m=center.tolist(),axis_W=axis.tolist(),joint_anchor_W_m=anchor.tolist(),joint_axis_W=ja.tolist(),geom_pos_in_wheel_body=m.geom_pos[gi].tolist(),geom_quat_in_wheel_body=m.geom_quat[gi].tolist(),center_to_ball_m=float(np.linalg.norm(center-ball)),nearest_axis_point_W_m=point.tolist(),nearest_axis_parameter_m=t,analytic_signed_gap_m=gap,joint_axis_to_geom_center_offset_m=float(np.linalg.norm(np.cross(center-anchor,ja))),abs_axis_alignment=float(abs(axis@ja))))
    return dict(ball_center_W_m=ball.tolist(),ball_radius_m=radius,wheels=wheels)

def contact_sample(env):
    m,d=env.model,env.data;nv=m.nv;net=np.zeros(nv);rows=[];ground_force=np.zeros(3);contact_power=0.
    for ci in range(d.ncon):
        c=d.contact[ci];g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2]);frame=c.frame.reshape(3,3).copy();force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
        jp1=np.zeros((3,nv));jr1=np.zeros((3,nv));jp2=np.zeros((3,nv));jr2=np.zeros((3,nv))
        mujoco.mj_jac(m,d,jp1,jr1,c.pos,b1);mujoco.mj_jac(m,d,jp2,jr2,c.pos,b2)
        J=jp2-jp1;Jr=jr2-jr1;v=J@d.qvel;omega=Jr@d.qvel;vc=frame@v;fw=frame.T@force[:3];tw=frame.T@force[3:]
        net+=J.T@fw+Jr.T@tw
        contact_power+=float(fw@v+tw@omega)
        n1,n2=m.geom(g1).name,m.geom(g2).name;wheel=None
        for k in range(3):
            if f'wheel_mesh_{k}' in [n1,n2] and 'the_ball' in [n1,n2]:wheel=k;break
        label=f'wheel_{wheel}' if wheel is not None else 'ground' if env.ground_geom in [g1,g2] and env.ball_geom in [g1,g2] else n1+'__'+n2
        fn=float(force[0]);mu=np.array(c.friction[:2]);util=float(np.linalg.norm(force[1:3]/(mu*fn))) if fn>1e-10 else 0.
        if label=='ground':ground_force+=fw if g1==env.ground_geom else -fw
        axis_torque=0.;roll_alignment=None;axis_alignment=None;normal_axis_torque=0.
        if wheel is not None:
            wi=m.geom(f'wheel_mesh_{wheel}').id;sign=1 if g2==wi else -1;axis=d.xaxis[env.wheel_ids[wheel]];lever=c.pos-d.xanchor[env.wheel_ids[wheel]]
            axis_torque=float(axis@(np.cross(lever,sign*fw)+sign*tw));normal_axis_torque=float(axis@np.cross(lever,sign*fn*frame[0]))
            rolling=np.cross(axis,lever);rolling-=frame[0]*(frame[0]@rolling)
            roll_alignment=float(abs(frame[2]@rolling)/max(np.linalg.norm(rolling),1e-15));capsule_axis=d.geom_xmat[wi].reshape(3,3)[:,2];axis_alignment=float(abs(frame[1]@capsule_axis))
        rows.append(dict(time_s=float(d.time),pair=label,geom1=n1,geom2=n2,active=bool(c.efc_address>=0),gap_m=float(c.dist),normal_force_N=fn,tangent1_force_N=float(force[1]),tangent2_force_N=float(force[2]),tangential_force_N=float(np.linalg.norm(force[1:3])),normal_relative_velocity_m_s=float(vc[0]),axial_low_mu_slip_m_s=float(vc[1]),rolling_high_mu_slip_m_s=float(vc[2]),tangential_slip_m_s=float(np.linalg.norm(vc[1:3])),friction_ellipse_utilization=util,friction_power_W=float(force[1:3]@vc[1:3]),mu1=float(mu[0]),mu2=float(mu[1]),condim=int(c.dim),pos_x_m=float(c.pos[0]),pos_y_m=float(c.pos[1]),pos_z_m=float(c.pos[2]),normal_x=float(frame[0,0]),normal_y=float(frame[0,1]),normal_z=float(frame[0,2]),axis_contact_torque_Nm=axis_torque,axis_normal_torque_Nm=normal_axis_torque,low_axis_alignment=axis_alignment,high_rolling_alignment=roll_alignment))
    iterations=int(np.sum(d.solver_niter));gradients=[];stride=len(d.solver)//len(d.solver_niter)
    for island,n in enumerate(d.solver_niter):
        if n:gradients.append(float(d.solver[island*stride+int(n)-1].gradient))
    return rows,dict(time_s=float(d.time),wheel_contact_count=sum(r['pair'].startswith('wheel_') and r['active'] for r in rows),ground_contact_count=sum(r['pair']=='ground' and r['active'] for r in rows),contact_generalized_force_max_abs_gap_native_units=float(np.max(abs(net-d.qfrc_constraint))),contact_power_gap_W=float(net@d.qvel-contact_power),solver_iterations=iterations,solver_normalized_gradient=max(gradients,default=0.),ground_force_W_N=ground_force.tolist(),actuator_generalized_wheel_torque_Nm=d.qfrc_actuator[env.wheel_dofs].tolist(),passive_wheel_torque_Nm=d.qfrc_passive[env.wheel_dofs].tolist(),constraint_wheel_torque_Nm=d.qfrc_constraint[env.wheel_dofs].tolist())
