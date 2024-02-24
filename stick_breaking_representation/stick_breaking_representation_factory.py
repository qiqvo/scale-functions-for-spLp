from typing import Type
from stick_breaking_representation.stick_breaking_representation import StickBreakingRepresentation


class StickBreakingRepresentationFactory(object):
    def __init__(self, sbr_type: Type[StickBreakingRepresentation], **args) -> None:
        self.args = args
        self.sbr_type = sbr_type
        
    def create(self, process: StickBreakingRepresentation) -> StickBreakingRepresentation:
        return self.sbr_type(process=process, **self.args)
    