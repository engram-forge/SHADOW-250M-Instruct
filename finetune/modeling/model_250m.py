import os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import common
from common import D,NL,NH,NKV,HD,FFNH,FPD, RMS,RVQ,Block,StructStep, requant, cs,ffn_activation


def mtp_slices(hidden,targets,horizon):
    """Return base and K=2 conditional-MTP alignment slices."""
    if horizon not in (1,2): raise ValueError("MTP horizon must be 1 or 2")
    pairs=[(hidden,targets)]
    if horizon==2 and hidden.shape[1]>1:
        pairs.append((hidden[:,:-1],targets[:,:-1],targets[:,1:]))
    return pairs


class MTPModule(nn.Module):
    """One token-conditioned residual MLP for the second-token proposal."""
    def __init__(s,dim):
        super().__init__(); hidden=dim//2
        s.norm=RMS(dim)
        s.down=RVQ(dim,hidden,32,1)
        s.up=RVQ(hidden,dim,32,1)
        # Keep the initial residual perturbation small without blocking gradients.
        nn.init.normal_(s.up.weight,mean=0.0,std=1e-3)
    def forward(s,state,previous_token_embedding):
        mixed=s.norm(state+previous_token_embedding)
        hidden=F.silu(s.down(ffn_activation(mixed)))
        return state+s.up(ffn_activation(hidden))


def load_vocab(fp_path,dev):
    fp=np.unpackbits(np.load(fp_path),axis=1)[:,:FPD]
    cent=torch.tensor(fp.astype(np.float32)*2-1,device=dev)      
    return cent, F.normalize(cent,dim=-1), cent.shape[0]

class Shadow250M(nn.Module):
    def __init__(s,cent,cent_n,V,use_memory=False,mtp_horizon=1):
        super().__init__()
        if int(mtp_horizon) not in (1,2): raise ValueError("MTP horizon must be 1 or 2")


        s.register_buffer("cent",cent,persistent=False)
        s.register_buffer("cent_n",cent_n,persistent=False); s.V=V
        s.inp=nn.Linear(FPD,D,bias=False)
        s.b=nn.ModuleList([Block(i) for i in range(NL)])         
        s.struct=StructStep()                                    
        s.nf=RMS(D); s.head=nn.Linear(D,FPD,bias=False)          
        s.mtp_horizon=int(mtp_horizon)
        s.mtp=MTPModule(D) if s.mtp_horizon==2 else None
        s.tied_bias=nn.Parameter(torch.zeros(V))


        s.default_use_memory=bool(use_memory)
        inv=1.0/(10000**(torch.arange(0,HD,2).float()/HD)); s.register_buffer("inv",inv,persistent=False)
    def trunk(s,idx):
        cos,sin=cs(idx.shape[1],idx.device); x=s.inp(s.cent[idx])
        for blk in s.b: x=blk(x,cos,sin)
        x,conf=s.struct(x)                                       
        return s.nf(x), conf
    def logits(s,x): return (s.head(x).float()@s.cent_n.T)+s.tied_bias      
    def _vocab_metrics(s,hidden,targets,head,chunk):
        projected=head(hidden).float().reshape(-1,FPD); flat=targets.reshape(-1); valid=flat>=0
        projected,flat=projected[valid],flat[valid]
        if not flat.numel():
            zero=hidden.float().sum()*0
            return zero,zero.detach(),0
        total=hidden.float().new_zeros(()); correct=hidden.new_zeros((),dtype=torch.long)
        for start in range(0,len(flat),chunk):
            logits=projected[start:start+chunk]@s.cent_n.T+s.tied_bias
            total+=F.cross_entropy(logits,flat[start:start+chunk],reduction="sum")
            correct+=logits.argmax(-1).eq(flat[start:start+chunk]).sum()
        return total/len(flat),correct.float()/len(flat),len(flat)
    def _vocab_loss(s,hidden,targets,head,chunk):
        return s._vocab_metrics(hidden,targets,head,chunk)[0]
    def mtp_hidden(s,hidden,previous_token_ids):
        if s.mtp is None: raise RuntimeError("model has no MTP module")
        if bool((previous_token_ids<0).any()):
            raise ValueError("MTP conditioning token IDs must be unmasked")
        token_embedding=s.inp(s.cent[previous_token_ids])
        return s.mtp(hidden,token_embedding)
    def mtp_logits(s,hidden,previous_token_ids):
        """Score the K=2 proposal conditioned on the accepted first token."""
        return s.logits(s.mtp_hidden(hidden,previous_token_ids))
    def language_model_losses(s,hidden,targets,conditioning_ids=None,chunk=2048):
        """Return one independently normalized loss for each available offset."""
        pairs=mtp_slices(hidden,targets,s.mtp_horizon)
        losses=[s._vocab_loss(pairs[0][0],pairs[0][1],s.head,chunk)]
        if len(pairs)>1:
            state,masked_previous,target=pairs[1]
            previous=(masked_previous if conditioning_ids is None
                      else conditioning_ids[:,:state.shape[1]])
            future=s.mtp_hidden(state,previous)
            losses.append(s._vocab_loss(future,target,s.head,chunk))
        return losses
    def language_model_metrics(s,hidden,targets,mtp_loss_weight=0.0,chunk=2048,
                               conditioning_ids=None):
        if mtp_loss_weight<0: raise ValueError("MTP loss weight must be nonnegative")
        base_loss,base_accuracy,base_tokens=s._vocab_metrics(
            hidden,targets,s.head,chunk)
        result={"loss":base_loss,"base_loss":base_loss,
                "base_accuracy":base_accuracy,"base_tokens":base_tokens,
                "mtp_loss_weight":float(mtp_loss_weight)}
        if s.mtp_horizon==2 and hidden.shape[1]>1:
            state=hidden[:,:-1]; target=targets[:,1:]
            previous=(targets[:,:-1] if conditioning_ids is None
                      else conditioning_ids[:,:state.shape[1]])
            future=s.mtp_hidden(state,previous)
            mtp_loss,mtp_accuracy,mtp_tokens=s._vocab_metrics(
                future,target,s.head,chunk)
            result.update(mtp_loss=mtp_loss,mtp_accuracy=mtp_accuracy,
                          mtp_tokens=mtp_tokens)
            result["loss"]=base_loss+mtp_loss_weight*mtp_loss
        else:
            zero=base_loss.detach()*0
            result.update(mtp_loss=zero,mtp_accuracy=zero,mtp_tokens=0)
        return result
    def language_model_loss(s,hidden,targets,mtp_loss_weight=0.0,chunk=2048,
                            conditioning_ids=None):
        if mtp_loss_weight==0:
            return s._vocab_loss(hidden,targets,s.head,chunk)
        return s.language_model_metrics(
            hidden,targets,mtp_loss_weight,chunk,conditioning_ids)["loss"]
    def forward(s,idx,ys=None):
        x,conf=s.trunk(idx)
        if ys is None: return s.logits(x)
        return s.language_model_loss(x,ys,conditioning_ids=idx[:,1:])

    @torch.no_grad()
    def prefill_cached(s,idx,max_ctx=2048,exact_shiftmax=True,
                       use_memory=None,memory_chunk=128,memory_capacity=256,
                       retrieval_trace=False,stream_block=256):
        if use_memory is None: use_memory=s.default_use_memory
        if idx.shape[1]>max_ctx:
            if not use_memory:
                idx=idx[:,-max_ctx:]
            else:
                logits,state=s.prefill_cached(
                    idx[:,:max_ctx],max_ctx=max_ctx,
                     exact_shiftmax=exact_shiftmax,use_memory=use_memory,
                     memory_chunk=memory_chunk,memory_capacity=memory_capacity,
                     retrieval_trace=retrieval_trace,stream_block=stream_block)
                cursor=max_ctx
                final=idx.shape[1]-1
                while cursor<final:
                    stop=min(final,cursor+int(stream_block))
                    state=s.ingest_cached(idx[:,cursor:stop],state)
                    cursor=stop
                return s.decode_cached(idx[:,final:final+1],state)
        cos,sin=cs(idx.shape[1],idx.device); x=s.inp(s.cent[idx]); layers=[]
        for blk in s.b:
            x,cache=blk.prefill_cached(x,cos,sin,max_ctx=max_ctx,
                                       exact=exact_shiftmax,
                                       archive_cold=bool(use_memory),
                                       cold_chunk=memory_chunk,
                                       retrieval_trace=retrieval_trace); layers.append(cache)
        trunk=x[:,-max_ctx:]; y,_=s.struct(x); ph=s.head(s.nf(y[:,-1])).float()
        return ph@s.cent_n.T+s.tied_bias,{"layers":layers,"trunk":trunk,
            "position":idx.shape[1],"max_ctx":max_ctx,
            "exact_shiftmax":exact_shiftmax,
            "retrieval_trace":bool(retrieval_trace),
            "stream_block":int(stream_block),
            "memory":common.make_memory_state(
                idx.shape[0],D,idx.device,chunk_size=memory_chunk,
                capacity=memory_capacity) if use_memory else None}

    @torch.no_grad()
    def ingest_cached(s,idx_block,state):
        if idx_block.shape[1]==0:
            return state
        x=s.inp(s.cent[idx_block]); pos=int(state["position"])
        for i,blk in enumerate(s.b):
            x,state["layers"][i]=blk.decode_cached(
                x,pos,state["layers"][i],max_ctx=state["max_ctx"],
                exact=state["exact_shiftmax"],retrieve_cold=False)
        combined=torch.cat((state["trunk"],x),1)
        overflow=max(0,combined.shape[1]-state["max_ctx"])
        common.memory_absorb_evicted(
            state.get("memory"),combined[:,:overflow],s.head,s.nf)
        state["trunk"]=combined[:,overflow:]
        s.struct.decode_cached(x,state["trunk"])
        state["position"]=pos+idx_block.shape[1]
        return state

    @torch.no_grad()
    def decode_cached(s,idx_new,state):
        if state.get("memory") is not None and idx_new.shape[1]!=1:
            logits=None
            for j in range(idx_new.shape[1]):
                logits,state=s.decode_cached(idx_new[:,j:j+1],state)
            return logits,state
        x=s.inp(s.cent[idx_new]); pos=int(state["position"])
        for i,blk in enumerate(s.b):
            x,state["layers"][i]=blk.decode_cached(x,pos,state["layers"][i],
                                                   max_ctx=state["max_ctx"],
                                                   exact=state["exact_shiftmax"])
        combined=torch.cat((state["trunk"],x),1)
        overflow=max(0,combined.shape[1]-state["max_ctx"])
        common.memory_absorb_evicted(
            state.get("memory"),combined[:,:overflow],s.head,s.nf)
        state["trunk"]=combined[:,overflow:]
        context=common.memory_append_recall(
            state.get("memory"),x[:,-1],state["trunk"],s.head,s.nf)
        y,_=s.struct.decode_cached(x,context); state["position"]=pos+idx_new.shape[1]
        ph=s.head(s.nf(y[:,-1])).float(); return ph@s.cent_n.T+s.tied_bias,state

if __name__=="__main__":
    dev="cuda" if torch.cuda.is_available() else "cpu"
    FP=os.path.join(os.path.dirname(__file__),"dolma3_fp_512b.npy")
    if not os.path.exists(FP): FP=os.path.join(os.path.dirname(__file__),"..","..","shadow-o","phase0_tokenizer","dolma3_fp_512b.npy")
    cent,cent_n,V=load_vocab(FP,dev); m=Shadow250M(cent,cent_n,V).to(dev)
    body=sum(p.numel() for n,p in m.named_parameters() if "cent" not in n and "tied_bias" not in n)
    print(f"SHADOW-O 250M | body {body/1e6:.1f}M params | vocab {V} (0 params) | config d{D} L{NL} GQA{NH}/{NKV}")
    b=m.b[0]; print(f"  proj {b.q.bits():.3f} bit/w | FFN {b.up.bits():.3f} bit/w | struct step: 1 | Q4/loop: none")
    x=torch.randint(0,V,(1,64),device=dev); requant(m)
    with torch.autocast("cuda",dtype=torch.bfloat16) if dev=="cuda" else torch.no_grad():
        lg=m(x); print(f"  forward OK: logits {tuple(lg.shape)} finite={bool(torch.isfinite(lg).all())}")
