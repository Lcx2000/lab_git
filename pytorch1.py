from __future__ import print_function
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
# x = torch.empty(5, 3)
# print(x)
# x = torch.rand(4, 4)
# print(x)
# print(x[:, 1])
# z = x.view(-1, 8) 
# print(z)
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
# x = torch.tensor([5.5, 3])
# print(x)

# def get_fake_data(batch_size=32):
#     ''' y=x*2+3 '''
#     x = torch.randn(batch_size, 1) * 20
#     y = x * 2 + 3 + torch.randn(batch_size, 1)
#     return x, y

# x, y = get_fake_data()

# class LinerRegress(torch.nn.Module):
#     def __init__(self):
#         super(LinerRegress, self).__init__()
#         self.fc1 = torch.nn.Linear(1, 1)

#     def forward(self, x):
#         return self.fc1(x)


# net = LinerRegress()
# loss_func = torch.nn.MSELoss()
# optimzer = optim.SGD(net.parameters())

# for i in range(40000):
#     optimzer.zero_grad()

#     out = net(x)
#     loss = loss_func(out, y)
#     loss.backward()

#     optimzer.step()

# w, b = [param.item() for param in net.parameters()]
# print (w, b)  # 2.01146, 3.184525

# # 显示原始点与拟合直线
# plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
# plt.plot(x.squeeze().numpy(), (x*w + b).squeeze().numpy())
# plt.show()
x = torch.randn((4,4), requires_grad=True)
y = 2*x
z = y.sum()

print (z.requires_grad)  # True

z.backward()

print(x.grad)
'''
tensor([[ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.]])
'''