"""
Base class for all Vision-Language Model wrappers.

All VLM implementations should inherit from LargeMultimodalModel and implement
the required methods for MTRE logit extraction.
"""


class LargeMultimodalModel:
    """
    Abstract base class for Vision-Language Model wrappers.

    All model implementations must inherit from this class and implement:
        - forward_with_probs(): For extracting logits from the first N tokens
        - get_p_true() (optional): For P(True) baseline computation

    Attributes:
        device (str): Device to run inference on ('cuda' or 'cpu')

    Example:
        >>> class MyVLM(LargeMultimodalModel):
        ...     def forward_with_probs(self, image, prompt):
        ...         # Implementation here
        ...         return response, output_ids, logits, probs
    """

    def __init__(self):
        """Initialize the base model with default device."""
        self.device = "cuda"

    def forward(self, image, prompt):
        """
        Basic forward pass (deprecated, use forward_with_probs instead).

        Args:
            image: Input image (numpy array, RGB format)
            prompt (str): Text prompt/question

        Returns:
            str: Model's text response
        """
        return ""

    def forward_with_probs(self, image, prompt):
        """
        Forward pass with probability/logit extraction.

        This is the main method used by MTRE for hallucination detection.
        Implementations should extract logits from the first N generated tokens.

        Args:
            image: Input image (numpy array, RGB format, H x W x 3)
            prompt (str): Text prompt/question for the model

        Returns:
            tuple: (response, output_ids, logits, probs)
                - response (str): Generated text response
                - output_ids (np.ndarray): Token IDs of generated tokens
                - logits (np.ndarray): Logit vectors, shape [num_tokens, vocab_size]
                - probs (np.ndarray): Probability vectors, shape [num_tokens, vocab_size]

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement forward_with_probs()")

    def get_p_true(self, image, prompt):
        """
        Compute P(True) baseline score.

        Used for the P(True) baseline method that asks the model
        whether its answer is correct.

        Args:
            image: Input image (numpy array, RGB format)
            prompt (str): Prompt asking if the answer is true/false

        Returns:
            float: Log probability of the model answering "True"

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_p_true()")