# nanoDeepSeek
A minimalistic DeepSeek-V2 like model with optimized training based on nanoGPT.

Which is then used to compare different positional embeddings 
e.g. Absolute positional embedding or rotary positional embedding (RoPE) as well as a combination of both 
together with different attention strategies e.g. Causal Self Attention and multi-head latent attention (MLA) 
with each other. 

The model is also used to test multiple configurations of DeepSeek-MoE, to give a back-to-back comparison 
between many small against only a few large shared experts, all while the overall number of model parameters 
and the number of parameters per forward pass remains unchanged.
