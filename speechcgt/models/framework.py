from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from ..graph.schema import GraphSchema
from ..typing import SpeechCGTOutput
from .classifier import MonotonicDomainClassifier
from .encoder import PGCGTEncoder
from .readout import CognitiveReadoutHead


class SpeechCGTFramework(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        graph_schema: GraphSchema,
        encoder: PGCGTEncoder | None = None,
        readout: CognitiveReadoutHead | None = None,
        classifier: MonotonicDomainClassifier | None = None,
    ) -> None:
        super().__init__()
        self.graph_schema = graph_schema
        self.encoder = encoder or PGCGTEncoder(config, graph_schema)
        self.readout = readout or CognitiveReadoutHead(config, graph_schema)
        self.classifier = classifier or MonotonicDomainClassifier(config)

    def forward(
        self,
        graph_batch: Batch,
        x_hat: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
        return_aux: bool = False,
    ) -> SpeechCGTOutput | tuple[SpeechCGTOutput, dict[str, torch.Tensor]]:
        hidden_states, final_attention, final_gates = self.encoder(graph_batch, x_hat, feature_mask)
        public_outputs, aux_outputs = self.readout(hidden_states, final_attention, final_gates, self.graph_schema)
        ci_logit, ci_prob = self.classifier(public_outputs["domain_scores"])
        output = SpeechCGTOutput(
            ci_logit=ci_logit,
            ci_prob=ci_prob,
            feature_scores=public_outputs["feature_scores"],
            function_scores=public_outputs["function_scores"],
            domain_scores=public_outputs["domain_scores"],
            path_fc=public_outputs["path_fc"],
            path_cd=public_outputs["path_cd"],
        )
        if return_aux:
            return output, aux_outputs
        return output

