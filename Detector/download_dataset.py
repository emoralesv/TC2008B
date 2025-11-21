from roboflow import Roboflow


rf = Roboflow(api_key="tOj9s7Mvp12RVGME97Nj")
project = rf.workspace("greenhouse-hnq7q").project("rugoso_test-mv4rs")
version = project.version(1)
dataset = version.download("yolov12")
                