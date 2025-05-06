import pickle

flie = open('/home/ubuntu518/桌面/ENERO/Enero_datasets/rwds-results-1-link_capacity-unif-05-1-zoo/SP_3top_15_B_NEW/Aarnet/Aarnet.48.pckl', 'rb')
info = pickle.load(flie)

print(info)