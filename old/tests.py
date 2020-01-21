import pytest
from eval_syntactic_diversity import _find_trigger

class TestEvalSyntacticDiversity():
    def test_wrap_trigger(self):
        sent = ["Her second husband, Philip M. M. Middleton Jr, also a doctor, died in 2009.",
                "In 1980, he married Kate Middleton."]
        required_annotated_sent = ["Her second [t husband] , Philip M. M. Middleton Jr , also a doctor , died in 2009 .", 
                                   "In 1980 , he [t married] Kate Middleton ."]
        annotated_sent = _find_trigger(sent)
        assert annotated_sent == required_annotated_sent