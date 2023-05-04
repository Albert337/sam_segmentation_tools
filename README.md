# sam_segmentation_tools
based on SAM model,generate the mask of roi 

## how to use the tools?
- first install the requirements.txt,like opencv,pyqt5,and other bags that sam need
- enter ui_test,and then run show_ui.py

## instructions
- 导入需要分割的图像或者文件夹
- 点击seg完成当前图像的分割
- 在左侧的label框中显示分割的效果，用不同的线条框选出来
- 在左侧的label中显示的分割结果，通过鼠标左侧选择目标区域，同时会在右侧显示出当前生成的mask．
- 鼠标左键在左侧label中不断选择会在右侧一直选择区域的mask（多次选择会合并显示），鼠标右键则会撤销上一次选择
- 当完成一个目标所有区域的选择后，点击finish，弹出一个对话框，输入标签完成对整个所选区域的标注
- 当当前图像所有想标注的目标完成后，点击save就会生成当前图像完整标注的mask．
