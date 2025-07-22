import os
import Sofa
import Sofa.Core
import Sofa.Simulation
from PluginList import pluginList
import numpy as np
import csv
import json

project_dir = os.path.dirname(os.path.abspath(__file__))
mesh_dir = os.path.join(project_dir, "meshes")
sensor_stl_mesh = os.path.join(mesh_dir, "DIGIT_v8.STL")
sensor_vtk_mesh = os.path.join(mesh_dir, "DIGIT_v8.vtk")

with open(os.path.join(project_dir, "Unseen_indenter_settings.json"), "r") as f:
    indenter_list = json.load(f)

def DIGITSensor(name="DIGITSensor",  scale=[1.0, 1.0, 1.0]):
    self = Sofa.Core.Node(name)
    mechanicalmodel = self.addChild("MechanicalModel")
    mechanicalmodel.addObject("MeshVTKLoader", name="loader", filename=sensor_vtk_mesh, scale3d=scale)
    mechanicalmodel.addObject("MeshTopology", src="@loader")
    mechanicalmodel.addObject("MechanicalObject", name="dofs", template="Vec3d")
    mechanicalmodel.addObject("UniformMass", totalMass=0.001)
    mechanicalmodel.addObject("TetrahedronFEMForceField", poissonRatio=0.41, youngModulus=3000)
    mechanicalmodel.addObject("CGLinearSolver", iterations=25, tolerance=1e-9, threshold=1e-9)
    mechanicalmodel.addObject("GenericConstraintSolver", maxIterations=500, tolerance=1e-8)
    mechanicalmodel.addObject("BoxROI", name="boxROI", box=[0.03, -0.005, 0.0005, -0.03, 0.03, -0.001])
    mechanicalmodel.addObject("RestShapeSpringsForceField", points="@boxROI.indices", stiffness=1e12, angularStiffness=1e12)
    mechanicalmodel.addObject("BoxROI", name="topROI", box=[0.01, 0.03, 0.0021, -0.01, -0.002, 0.00], drawBoxes=False)
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

    return self

def Indenter(info):
    name = info["name"]
    translation = [p / 1000.0 for p in info["translation"]]
    #scale = info["scale"]
    scale = [1, 1,1]
    rotation = info.get("orientation", [0, 0, 0])

    indenter_stl = os.path.join(mesh_dir, info["filename_stl"])
    indenter_vtk = os.path.join(mesh_dir, info["filename_vtk"])

    self = Sofa.Core.Node(name)
    mechanicalmodel = self.addChild("MechanicalModel")
    mechanicalmodel.addObject("MeshVTKLoader", name="meshLoader", filename=indenter_vtk, translation=translation, rotation=rotation, scale3d=scale)
    mechanicalmodel.addObject("MeshTopology", src="@meshLoader")
    mechanicalmodel.addObject("MechanicalObject", template="Vec3d", name="dofs", position="@meshLoader.position")
    mechanicalmodel.addObject("UniformMass", totalMass=0.01)

    collision = mechanicalmodel.addChild("CollisionModel")
    collision.addObject("MeshSTLLoader", name="collisionLoader", filename=indenter_stl, translation=translation, rotation=rotation, scale3d=scale)
    collision.addObject("MeshTopology", src="@collisionLoader")
    collision.addObject("MechanicalObject", name="collision_dofs", template="Vec3d")
    collision.addObject("TriangleCollisionModel", simulated=True, moving=True)
    collision.addObject("PointCollisionModel", simulated=True, moving=True)
    collision.addObject("BarycentricMapping", input="@../dofs", output="@collision_dofs")

    visual = mechanicalmodel.addChild("VisualModel")
    visual.addObject("MeshSTLLoader", name="loader", filename=indenter_stl, translation=translation, rotation=rotation, scale3d=scale)
    visual.addObject("OglModel", src="@loader", color=[0.5, 0.5, 0.5, 0.5])
    visual.addObject("BarycentricMapping", input="@../dofs", output="@.")

    return self

class LinearMotionControllerWithDepth(Sofa.Core.Controller):
    def __init__(self, node, indenter_name, axis=[0, 0, 1], speed=0.00001, depth=-0.001, save_interval=1):
        super().__init__()
        self.node = node
        self.indenter_name = indenter_name
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

        if self.step % self.save_interval == 0:
            try:
                root = self.node.getRoot()
                digit = root.getChild("DIGITSensor")
                mech = digit.getChild("MechanicalModel")
                top_roi = mech.getObject("topROI")
                indices = top_roi.findData("indices").value
                dofs = mech.getObject("dofs")
                positions = np.array(dofs.findData("position").value)
                rest_positions = np.array(dofs.findData("rest_position").value)

                rows = [[*positions[i], rest_positions[i][2] - positions[i][2]] for i in indices]

                base_dir = os.path.join(project_dir, "NodalDataOutput", self.indenter_name)
                os.makedirs(base_dir, exist_ok=True)
                filename = os.path.join(base_dir, f"topROI_step_{self.step-1:03d}.csv")

                with open(filename, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["x", "y", "z", "dz"])
                    writer.writerows(rows)

                print(f"[INFO] Saved {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save deformation data: {e}")

        self.current_pos += self.speed * self.direction
        self.dofs.findData("position").value = self.current_pos.tolist()

def createScene(rootNode, info):
    rootNode.gravity = [0.0, 0.0, -9.81]
    rootNode.dt = 0.01
    rootNode.addObject("RequiredPlugin", pluginName=pluginList)
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject("SparseLDLSolver", name="solver", template="CompressedRowSparseMatrixMat3x3d")
    rootNode.addObject("NewmarkImplicitSolver", gamma=0.5, beta=0.25, rayleighStiffness=0.01, rayleighMass=0.01)
    rootNode.addObject("DefaultPipeline")
    rootNode.addObject("BruteForceBroadPhase")
    rootNode.addObject("BVHNarrowPhase")
    rootNode.addObject("CGLinearSolver", iterations=100, tolerance=1e-8, threshold=1e-9)
   
    rootNode.addObject("LocalMinDistance", alarmDistance=5e-5, contactDistance=3e-6, angleCone=0.0)
    rootNode.addObject("GenericConstraintSolver", tolerance=1e-5, maxIterations=500)
    rootNode.addObject("GenericConstraintCorrection")
    rootNode.addObject("DefaultContactManager", response="FrictionContactConstraint", responseParams="mu=0.2 contactStiffness=30 ")
    rootNode.addObject("VisualStyle", displayFlags="showForceFields")
    rootNode.addObject("BackgroundSetting", color=[0, 0, 0, 1])

    sensor = rootNode.addChild(DIGITSensor())
    indenter = rootNode.addChild(Indenter(info))
    indenter.addObject(LinearMotionControllerWithDepth(indenter, info["name"], axis=[0, 0, 1], speed=0.00001, depth=-0.001))

    return rootNode

def run_all():
    for info in indenter_list:
        print(f"\n[INFO] Running simulation for indenter: {info['name']}")
        root = Sofa.Core.Node("root")
        createScene(root, info)
        Sofa.Simulation.init(root)
        for step in range(100):
            Sofa.Simulation.animate(root, root.dt.value)
        Sofa.Simulation.unload(root)

if __name__ == '__main__':
    run_all()
