import os
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv
import Sofa
import Sofa.Gui
import Sofa.Core
import Sofa.Simulation
from PluginList import pluginList

project_dir = os.path.dirname(os.path.abspath(__file__))
mesh_dir = os.path.join(project_dir, "meshes")
sensor_stl_mesh = os.path.join(mesh_dir, "DIGIT_remesh2.STL")
sensor_vtk_mesh = os.path.join(mesh_dir, "DIGIT_remesh2.vtk")
sensor_body_stl_mesh = os.path.join(mesh_dir, "DIGIT_body.STL")
# indenter_stl_mesh = os.path.join(mesh_dir, "Sphere_object.STL") # Indenter/square
# indenter_vtk_mesh = os.path.join(mesh_dir, "Sphere_object.vtk")
indenter_stl_mesh = os.path.join(mesh_dir, "Indenter/round.STL") # Indenter/square
indenter_vtk_mesh = os.path.join(mesh_dir, "Indenter/round.vtk")

#Pos_center = [0.594, 11.750, 18] # Sphere Ball

Pos_center = [0.0, 11.750, 4]
indenterPos = [Pos_center[i]/1000 for i in range(3)]




def DIGITSensor(name="DIGITSensor", scale=[1.2, 1.2, 1.0]):
    self = Sofa.Core.Node(name)
    mechanicalmodel = self.addChild("MechanicalModel")
    mechanicalmodel.addObject("MeshVTKLoader", name="loader", filename=sensor_vtk_mesh, scale3d=scale)
    mechanicalmodel.addObject("MeshTopology", src="@loader")
    mechanicalmodel.addObject("MechanicalObject", name="dofs", template="Vec3d")
    mechanicalmodel.addObject("UniformMass", totalMass=0.001)
    mechanicalmodel.addObject("TetrahedronFEMForceField", poissonRatio=0.41, youngModulus=30000)
    mechanicalmodel.addObject("CGLinearSolver", iterations=25, tolerance=1e-9, threshold=1e-9)
    mechanicalmodel.addObject("GenericConstraintSolver", maxIterations=500, tolerance=1e-8)
    mechanicalmodel.addObject("BoxROI", name="boxROI", box=[0.03, -0.005, 0.0005, -0.03, 0.03, -0.001], drawBoxes=False)
    mechanicalmodel.addObject("RestShapeSpringsForceField", points="@boxROI.indices", stiffness=1e12, angularStiffness=1e12)
    mechanicalmodel.addObject("BoxROI", name="topROI", box=[0.01, 0.03, 0.0041, -0.01, -0.002, 0.000], drawBoxes=False)
    mechanicalmodel.addObject("EulerImplicitSolver", name="dampingSolver", rayleighStiffness=0.002, rayleighMass=0.001)

    collision = mechanicalmodel.addChild("CollisionModel")
    collision.addObject("MeshSTLLoader", name="loader", filename=sensor_stl_mesh, scale3d=scale)
    collision.addObject("MeshTopology", src="@loader")
    collision.addObject("MechanicalObject", name="collision_dofs")
    collision.addObject("TriangleCollisionModel", simulated=True, selfCollision=False)
    collision.addObject("PointCollisionModel", simulated=True)
    collision.addObject("BarycentricMapping", input="@../dofs", output="@collision_dofs")

    visual = mechanicalmodel.addChild("VisualModel")
    visual.addObject("MeshSTLLoader", name="loader", filename=sensor_stl_mesh, scale3d=scale)
    visual.addObject("OglModel", src="@loader", color=[0.8, 0.8, 0.8, 0.8])
    visual.addObject("BarycentricMapping")

    sensor_body_visual = mechanicalmodel.addChild("BodyVisual")
    sensor_body_visual.addObject("MeshSTLLoader", name="bodyLoader", filename=sensor_body_stl_mesh, translation=[0, -0.004, -0.028], scale3d=[1, 1, 1])
    sensor_body_visual.addObject("OglModel", src="@bodyLoader", color=[0.3, 0.3, 0.3, 1.0])

    return self

def Indenter(name="Indenter", translation=indenterPos, rotation=[0, 0, 0], scale=[0.001, 0.001, 0.001]): #scale=[0.001, 0.001, 0.001]
    self = Sofa.Core.Node(name)
    mechanicalmodel = self.addChild("MechanicalModel")
    mechanicalmodel.addObject("MeshVTKLoader", name="meshLoader", filename=indenter_vtk_mesh, translation=translation, rotation=rotation, scale3d=scale)
    mechanicalmodel.addObject("MeshTopology", src="@meshLoader")
    mechanicalmodel.addObject("MechanicalObject", template="Vec3d", name="dofs", position="@meshLoader.position")
    mechanicalmodel.addObject("UniformMass", totalMass=0.01)

    collision = mechanicalmodel.addChild("CollisionModel")
    collision.addObject("MeshSTLLoader", name="collisionLoader", filename=indenter_stl_mesh, translation=translation, rotation=rotation, scale3d=scale)
    collision.addObject("MeshTopology", src="@collisionLoader")
    collision.addObject("MechanicalObject", name="collision_dofs", template="Vec3d")
    collision.addObject("TriangleCollisionModel", simulated=True, moving=True)
    collision.addObject("PointCollisionModel", simulated=True, moving=True)
    collision.addObject("BarycentricMapping", input="@../dofs", output="@collision_dofs")

    visual = mechanicalmodel.addChild("VisualModel")
    visual.addObject("MeshSTLLoader", name="loader", filename=indenter_stl_mesh, translation=translation, rotation=rotation, scale3d=scale)
    visual.addObject("OglModel", src="@loader", color=[0.9, 0.3, 0.3, 1])
    visual.addObject("BarycentricMapping", input="@../dofs", output="@.")

    return self
class LinearMotionControllerWithDepth(Sofa.Core.Controller):
    def __init__(self, node, axis=[0, 0, 1], speed=0.0001, depth=-0.002, save_interval=1):
        super().__init__()
        self.node = node
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.speed = abs(speed)
        self.depth = abs(depth)
        self.save_interval = save_interval
        self.model = node.getChild("MechanicalModel")
        self.dofs = self.model.getObject("dofs")
        self.init_pos = np.array(self.dofs.findData("position").value)
        self.current_pos = self.init_pos.copy()
        self.rest_pos = self.init_pos.copy()
        self.direction = self.axis * (-1 if depth < 0 else 1)
        self.target_pos = self.init_pos + self.direction * self.depth
        self.active = True
        self.step = 0

    def onAnimateBeginEvent(self, event):
        if not self.active:
            return

        self.step += 1
        tip_initial = self.init_pos[0]
        tip_current = self.current_pos[0]
        displacement_scalar = np.dot(tip_current - tip_initial, self.direction)

        if displacement_scalar >= self.depth - 1e-6:
            print("[INFO] Indenter reached target depth. Motion stopped.")
            self.active = False
            return

        # Save every N steps
        if self.step % self.save_interval == 0:
            try:
                root = self.node.getRoot()
                digit = root.getChild("DIGITSensor")
                mech = digit.getChild("MechanicalModel")
                dofs = mech.getObject("dofs")
                positions = np.array(dofs.findData("position").value)
                rest_positions = np.array(dofs.findData("rest_position").value)

                # rows = [[*p, p0[2] - p[2]] for p, p0 in zip(positions, rest_positions)]
                top_roi = mech.getObject("topROI")
                indices = top_roi.findData("indices").value
                rows = [[*positions[i], rest_positions[i][2] - positions[i][2]] for i in indices]

                output_dir = os.path.join(project_dir, "NodalDataOutput_ex")
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"topROI_step_{self.step-1:03d}.csv")
                with open(filename, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["x", "y", "z", "dz"])
                    writer.writerows(rows)
                print(f"[INFO] Saved {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save deformation data: {e}")

        self.current_pos += self.speed * self.direction
        self.dofs.findData("position").value = self.current_pos.tolist()

def createScene(rootNode ):
    rootNode.gravity = [0.0, 0.0, 0.0]
    rootNode.dt = 0.01
    rootNode.addObject("RequiredPlugin", pluginName=pluginList)
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject("SparseLDLSolver", name="solver", template="CompressedRowSparseMatrixMat3x3d")
    rootNode.addObject("NewmarkImplicitSolver", gamma=0.5, beta=0.25, rayleighStiffness=0.01, rayleighMass=0.01)
    rootNode.addObject("DefaultPipeline")
    rootNode.addObject("BruteForceBroadPhase")
    rootNode.addObject("BVHNarrowPhase")
    rootNode.addObject("CGLinearSolver", iterations=100, tolerance=1e-8, threshold=1e-9)
    rootNode.addObject("LocalMinDistance", alarmDistance=1e-4, contactDistance=5e-5, angleCone=0.0)

    rootNode.addObject("GenericConstraintSolver", tolerance=1e-5, maxIterations=500)
    rootNode.addObject("GenericConstraintCorrection")
    rootNode.addObject("DefaultContactManager", response="FrictionContactConstraint", responseParams="mu=0.2 contactStiffness=30 ")
    rootNode.addObject("VisualStyle", displayFlags="showForceFields")
    rootNode.addObject("BackgroundSetting", color=[1, 1, 1, 1])

    sensor = rootNode.addChild(DIGITSensor())
    indenter = rootNode.addChild(Indenter())
    indenter.addObject(LinearMotionControllerWithDepth(indenter, axis=[0, 0, 1], speed=0.00001, depth=-0.001))
    
    return rootNode

def run_sofa():
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("DIGITSim", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(800, 600)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == '__main__':
    run_sofa()
