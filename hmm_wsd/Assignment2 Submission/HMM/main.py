#!/usr/bin/env python
# coding: utf-8

# In[4]:


print("######################################################")
print("###############  WELCOME TO HMM MENU  ################")
print("######################################################")
print("YOU CAN CHOOSE FROM THE FOLLOWING OPTIONS:")
print("1.HMM based WSD including Untagged words(Accuracy: 75.99%) \nNote: Since unseen word is set to automatically tagged as untagged and 5/7 of words in corpus are untagged it shows this accuracy")
print("\n\n2.HMM based WSD excluding Untagged words with MFS for unseen tokens in Viterbi")
print("\n\n3.HMM based WSD excluding Untagged words with WFS for unseen tokens in Viterbi")
print("Select your option: ")
option_choosed=int(input())
if option_choosed==1:
    import hmm_wsd_incl_untagged
    exec('hmm_wsd_incl_untagged')
elif option_choosed==2:
    import hmm_plus_MFS
    exec('hmm_plus_MFS')
elif option_choosed==3:
    import hmm_plus_WFS
    exec('hmm_plus_WFS')
else:
    print('!!!!INVALID OPTION CHOOSEN!!!!')
    print('EXITING')
    


# In[ ]:




