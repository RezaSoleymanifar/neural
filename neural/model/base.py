"""
This module contains the base class for all models.
"""

class AbstractModel:
    """
    This is the base class for all models.
    """

    def __init__(self):
        """
        Initialize the model.
        """
        pass

    def __call__(self, observation):
        """
        Given an observation, return an array of actions.

        args:
        ----------
        observation (numpy.ndarray): 
            The observation from the environment.

        Returns:
        ----------
        numpy.ndarray: 
            An array of actions.
        """
        raise NotImplementedError

    def save(self, file_path):
        """
        Save the model to a file.

        Parameters:
        ----------
        file_path (str): 
            Path to save the model.
        """
        raise NotImplementedError
