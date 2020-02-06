import torch
from torch.nn import CrossEntropyLoss

class RCTransformer(torch.nn.Module):
    """
    This is Asaf's implementation of Matching the Blanks Paper.
    """

    def __init__(self, base_model, num_labels, class_weights=None):
        super(RCTransformer, self).__init__()
        self.num_labels = num_labels

        self.transformer = base_model
        if 'd_model' in vars(base_model.config):
            hidsize = base_model.config.d_model
        else:
            hidsize = base_model.config.hidden_size
        self.logits_proj = torch.nn.Linear(hidsize * 2, num_labels)
        self.class_weights = class_weights

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None,
                labels=None,
                markers_mask=None):

        assert markers_mask is not None

        transformer_outputs = self.transformer(input_ids, token_type_ids=token_type_ids,
                                               attention_mask=attention_mask,
                                               )
        output = transformer_outputs[0]

        batched_cat = []
        entstarts = []
        for i in range(markers_mask.size(0)):
            entstarts.append([i for i, v in enumerate(markers_mask[i]) if v])
        for i, (start_ent1, start_ent2) in enumerate(entstarts):
            batched_cat.append(output[i, [start_ent1, start_ent2]].view(-1))
        output = torch.stack(batched_cat)

        logits = self.logits_proj(output)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss(self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
