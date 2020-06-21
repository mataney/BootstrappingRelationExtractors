import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

class RobertaForRelationClassification(BertPreTrainedModel):
    """
    This class is similar to RobertaForSequenceClassification only we are using our own classifier
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        if os.environ['THIN_CLASSIFIER'] == '1':
            self.classifier = MTBClassificationHeadThin(config)
        else:
            self.classifier = MTBClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        markers_mask=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, markers_mask)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    

class MTBClassificationHead(nn.Module):
    """
    This is similar to MTBClassificationHead only taking the relevant markers
    instead of the <s> token.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, markers_mask):
        batch_size, _, feature_size = features.size()
        assert all(markers_mask.sum(1) == 2)
        # take [E1] and [E2] tokens
        x = features.masked_select(markers_mask.unsqueeze(2)).view(batch_size, 2*feature_size)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
  
class MTBClassificationHeadThin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size*2, config.num_labels)

    def forward(self, features, markers_mask):
        batch_size, _, feature_size = features.size()
        assert all(markers_mask.sum(1) == 2)
        # take [E1] and [E2] tokens
        x = features.masked_select(markers_mask.unsqueeze(2)).view(batch_size, 2*feature_size)
        x = self.out_proj(x)
        return x
  