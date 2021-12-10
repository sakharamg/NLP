#!/usr/bin/env python
# coding: utf-8

# In[4]:


print("######################################################")
print("###############  WELCOME TO HMM MENU  ################")
print("######################################################")
print("YOU CAN CHOOSE FROM THE FOLLOWING OPTIONS:")
print("1.HMM based WSD including Untagged words(Accuracy: 85.27%)")
print("2.HMM based WSD excluding Untagged words(Accuracy: 50.14%)")
print("Select your option: ")
option_choosed=int(input())
if option_choosed==1:
    import hmm_wsd_incl_untagged
    exec('hmm_wsd_incl_untagged')
elif option_choosed==2:
    import hmm_wsd_excluding_untagged
    exec('hmm_wsd_excluding_untagged')
else:
    print('!!!!INVALID OPTION CHOOSEN!!!!')
    print('EXITING')
    


# In[ ]:




