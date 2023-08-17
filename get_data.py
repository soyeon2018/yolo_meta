from roboflow import Roboflow
rf = Roboflow(api_key="VVPaeuzHP4i3na741qkt")
project = rf.workspace("roboflow-gw7yv").project("fish-yzfml")
dataset = project.version(44).download("yolov8")
