#TODO:resample之前需要把mask膨胀

#调参注意事项
1. ToTensor是会改变图像的维度的，所以在Totensor后要马上用permute把维度换回来。另外在用simpleITK读图时，其GetArrayFromImage函数也会将CT
图像的depth维度移到第一维。