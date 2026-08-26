"""Slow, exact greedy K=2 MTP proposal and sequential-verification reference."""

from dataclasses import dataclass

import torch


def _last_greedy(logits):
    return logits[:,-1].argmax(-1)


@dataclass
class K2Step:
    candidate_first: torch.Tensor
    candidate_second: torch.Tensor
    reference_first: torch.Tensor
    reference_second: torch.Tensor
    first_accepted: torch.Tensor
    second_accepted: torch.Tensor


@torch.no_grad()
def reference_k2_step(model,context):
    """Propose K=2, then verify with two ordinary greedy model forwards.

    This intentionally does not use the KV cache or fused verification. It is an
    executable definition of the semantics expected from a future native engine.
    """
    if getattr(model,"mtp_horizon",None)!=2 or getattr(model,"mtp",None) is None:
        raise ValueError("K=2 reference decoding requires a horizon-two MTP model")
    if context.ndim!=2 or context.shape[1]<1:
        raise ValueError("context must have shape (batch, sequence) with sequence >= 1")

    proposal_hidden,_=model.trunk(context)
    final_hidden=proposal_hidden[:,-1:]
    candidate_first=_last_greedy(model.logits(final_hidden))
    candidate_second=_last_greedy(
        model.mtp_logits(final_hidden,candidate_first[:,None]))

    # Deliberately recompute both positions through the ordinary base path.
    reference_first=_last_greedy(model(context))
    after_first=torch.cat((context,reference_first[:,None]),dim=1)
    reference_second=_last_greedy(model(after_first))

    first_accepted=candidate_first.eq(reference_first)
    second_accepted=(first_accepted & candidate_second.eq(reference_second))
    return K2Step(candidate_first,candidate_second,reference_first,reference_second,
                  first_accepted,second_accepted)


class AcceptanceMetrics:
    def __init__(self):
        self.cycles=0
        self.first_proposals=0
        self.first_accepted=0
        self.second_eligible=0
        self.second_accepted=0

    def update(self,step,second_eligible=None):
        first=step.first_accepted.detach().bool()
        second=step.second_accepted.detach().bool()
        eligible=(first if second_eligible is None else
                  second_eligible.detach().bool() & first)
        self.cycles+=1
        self.first_proposals+=first.numel()
        self.first_accepted+=int(first.sum())
        self.second_eligible+=int(eligible.sum())
        self.second_accepted+=int((second & eligible).sum())

    def as_dict(self):
        first_rate=self.first_accepted/max(1,self.first_proposals)
        second_rate=self.second_accepted/max(1,self.second_eligible)
        return {
            "cycles":self.cycles,
            "sequences":self.first_proposals,
            "first_accepted":self.first_accepted,
            "first_acceptance_rate":first_rate,
            "second_eligible":self.second_eligible,
            "second_accepted":self.second_accepted,
            "second_acceptance_rate":second_rate,
            "pair_acceptance_rate":self.second_accepted/max(1,self.first_proposals),
            "mean_accepted_draft_tokens":(self.first_accepted+self.second_accepted)
                                         /max(1,self.first_proposals),
        }


@torch.no_grad()
def reference_generate(model,context,cycles):
    """Generate the ordinary greedy sequence while measuring K=2 proposals.

    Oracle tokens are appended after every comparison, so later contexts remain
    identical to ordinary sequential greedy decoding even after an MTP rejection.
    """
    if cycles<0: raise ValueError("cycles must be nonnegative")
    output=context.clone(); metrics=AcceptanceMetrics(); steps=[]
    for _ in range(cycles):
        step=reference_k2_step(model,output); metrics.update(step); steps.append(step)
        output=torch.cat((output,step.reference_first[:,None],
                          step.reference_second[:,None]),dim=1)
    return output,metrics.as_dict(),steps
