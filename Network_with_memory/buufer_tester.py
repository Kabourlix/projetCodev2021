import dataExtractor as de



test = TrajDataSet(a,mem_nb=3)
test_loader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=True)
state = next(iter(test_loader))
print(f'Voici notre état {state} ainsi que sa shape.')