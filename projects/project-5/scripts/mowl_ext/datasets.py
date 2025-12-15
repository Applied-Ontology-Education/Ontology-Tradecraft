import logging

from mowl.datasets import PathDataset
from mowl.datasets.base import OWLClasses

from util.logger_config import config

logger = logging.getLogger(__name__)

config(logger)


class OtcPathDataset(PathDataset):
    """
    Project-specific PathDataset that overrides `evaluation_classes`.

    In mOWL, Dataset.evaluation_classes must return a pair (tuple) of OWLClasses
    collections to define the candidate space used during evaluation (e.g., for
    link prediction over subclass axioms). The base class leaves this abstract.

    Default behavior here:
    - Use the full set of classes gathered from training/validation/testing
      ontologies for both elements of the pair. This mirrors the typical setup
      in taxonomy link prediction where any class can be a potential head/tail.

    You can later specialize this to restrict to a subset (e.g., only classes
    appearing in simple named rdfs:subClassOf axioms) without changing callers.
    """

    def __init__(self, *args, eval_scope: str = "all", **kwargs):
        # eval_scope controls which class space is exposed via evaluation_classes
        # values: 'all' (default) -> train+valid+test
        #         'validation'     -> only classes from validation ontology (if present)
        #         'train'          -> only classes from training ontology
        self._eval_scope = eval_scope.lower() if isinstance(eval_scope, str) else "all"
        super().__init__(*args, **kwargs)

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            scope = self._eval_scope
            try:
                if scope == "validation" and self.validation is not None:
                    # Use only classes seen in the validation ontology
                    val_classes = list(self.validation.getClassesInSignature())
                    cls = OWLClasses(val_classes)
                    self._evaluation_classes = (cls, cls)
                    logger.info("evaluation_classes scope=validation (|C|=%d)", len(cls))
                elif scope == "train":
                    # Use only classes seen in the training ontology
                    tr_classes = list(self.ontology.getClassesInSignature())
                    cls = OWLClasses(tr_classes)
                    self._evaluation_classes = (cls, cls)
                    logger.info("evaluation_classes scope=train (|C|=%d)", len(cls))
                else:
                    # Default: full classes across train/valid/test via Dataset.classes
                    cls = self.classes
                    self._evaluation_classes = (cls, cls)
                    logger.info("evaluation_classes scope=all (|C|=%d)", len(cls))
            except Exception as e:
                # Fallback to full class set if anything goes wrong
                logger.warning("Falling back to scope=all for evaluation_classes due to: %s", e)
                cls = self.classes
                self._evaluation_classes = (cls, cls)
            logger.debug(
                "Initialized evaluation_classes with %d classes (pair).", len(self._evaluation_classes[0])
            )
        return self._evaluation_classes
