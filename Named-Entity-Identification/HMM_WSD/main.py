#!/usr/bin/env python
# coding: utf-8

# In[4]:


print("######################################################")
print("###############  WELCOME TO HMM MENU  ################")
print("######################################################")
print("YOU CAN CHOOSE FROM THE FOLLOWING OPTIONS:")
print("1.HMM based WSD with MFS")
print("\n\n2.HMM based WSD with WFS")
print("Select your option: ")
option_choosed=int(input())
if option_choosed==1:
    import hmm_plus_MFS
    exec('hmm_plus_MFS')
elif option_choosed==2:
    import hmm_plus_WFS
    exec('hmm_plus_WFS')
else:
    print('!!!!INVALID OPTION CHOOSEN!!!!')
    print('EXITING')
    


# In[ ]:




