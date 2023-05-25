from typing import Type, Dict, Optional

import numpy as np
from gym import (Env, Wrapper, ObservationWrapper, spaces, Space)

from neural.common.constants import ACCEPTED_OBSERVATION_TYPES, GLOBAL_DATA_TYPE
from neural.common.exceptions import IncompatibleWrapperError
from neural.wrapper.base import metadata
from neural.utils.base import FillDeque, RunningStatistics


def validate_observation(
        wrapper: Wrapper,
        observation: np.ndarray | Dict[str, np.ndarray]) -> None:
    """
    This is a helper function that is shared between the observation sanity checkers. It performs a
    basic check to see if the observation is a known observation type accepted by this library.

    Parameters
    ----------
    wrapper : Wrapper
        The wrapper object that received the observation.
    observation : np.ndarray or Dict[str, np.ndarray]
        The observation to be validated. If it is a dictionary, 
        all its values must be numpy arrays.

    Raises
    ------
    IncompatibleWrapperError
        If the wrapper received an observation of an incompatible type.

    """

    # Checks that the wrapper is compatible with the observation types. This is a
    #  helper function to avoid having to reimplement it
    if isinstance(observation, dict):
        if all(isinstance(observation[key], np.ndarray) for key in observation):
            valid = True

    elif isinstance(observation, np.ndarray):
        valid = True

    if not valid:
        raise IncompatibleWrapperError(
            f'Wrapper {type(wrapper).__name__} received an observation of type {type(observation)}, '
            F'which is not in the accepted observation types {ACCEPTED_OBSERVATION_TYPES}.'
        )

    return False


def observation(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:
    """
    A decorator that augments an existing Gym wrapper and extends it to have input and output self checks.
    The reulst is a new observation wrapper that checks if the observation of the wrapped env is in its expected observation space, and also
    check if the observation returned by the wrapped env is in its defined observation space.

    Parameters
    ----------
    wrapper_class : Type[Wrapper]
        The base Gym wrapper class to be augmented. 
        This should be a subclass of `gym.Wrapper`.

    Raises
    ------
    TypeError
        If the `wrapper_class` argument is not a subclass of `gym.Wrapper`.
    IncompatibleWrapperError
        If the observation space is not defined in the enclosed environment, 
        or if the expected observation type is not valid.

    Returns
    -------
    Type[Wrapper]
        A new wrapper class that checks if an observation is in the observation 
        space before returning it from the reset and step functions.

    Examples
    --------
    >>> from gym import Wrapper
    >>> from neural.meta.env.wrapper.observation import observation
    >>> @observation
    ... class CustomObservationWrapper(ObservationWrapper):
    ...     self.expected_observation_type = [np.ndarray, dict]
    ...     def observation(self, observation):
    ...         pass
    """

    if not issubclass(wrapper_class, ObservationWrapper):
        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {ObservationWrapper}"
        )

    class ObservationSpaceCheckerWrapper(wrapper_class):
        """
        This wrapper allows an observation wrapper to have input and output self checks by subclassing
        from it and overriding the observation method. The reulst is a new observation wrapper that
        checks if the observation of the wrapped env is in its expected observation space, and also
        check if the observation returned by the wrapped env is in its defined observation space.

        Parameters
        ----------
        env : gym.Env
            The environment being wrapped.
        *args : tuple
            Optional arguments to pass to the wrapper.
        **kwargs : dict
            Optional keyword arguments to pass to the wrapper.

        Raises
        ------
        IncompatibleWrapperError
            If the observation is not in the observation space.
    
        Methods
        -------
        __init__(self, env: Env, *args, **kwargs) -> None:
            Initializes the ObservationSpaceCheckerWrapper instance.
        _validate_expected_observation_type(self) -> None:
            Validates the expected observation type of the wrapper.
        _validate_observation_in_expected_observation_type(self, 
            observation: Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
            ) -> None:
            Validates if the observation is in the expected observation type.
        _validate_observation_in_observation_space(self,
            observation: Union[np.ndarray[float], Dict[str, np.ndarray[float]]]) -> None:
            Validates if the observation is in the observation space.
        observation(self, observation: Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
            ) -> Union[np.ndarray[float], Dict[str, np.ndarray[float]]]:
            Checks if the observation is in the observation space before returning it
            from the observation method of the base class.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:
            """
            Initializes the ObservationSpaceCheckerWrapper instance.

            Parameters
            ----------
            env : Env
                The environment being wrapped.
            *args : tuple
                Optional arguments to pass to the wrapper.
            **kwargs : dict
                Optional keyword arguments to pass to the wrapper.

            Raises
            ------
            IncompatibleWrapperError
                If the observation is not in the observation space.
            """

            # as a byproduct of following and due to inheritance from gym.ObservationWrapper
            # existence of self.observation_space is guaranteed
            if not hasattr(env, 'observation_space') or not isinstance(
                    env.observation_space, Space):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} requires a non None observation '
                    f'space of type {Space} to be defined in the enclosed environment '
                    f'{type(env).__name__}.')

            super().__init__(env, *args, **kwargs)

            self._validate_expected_observation_type()
            self._validate_observation_in_expected_observation_type(
                env.observation_space.sample())

            return None

        def _validate_expected_observation_type(self):
            """
            Validates the expected observation type of the wrapper. ACCEPTED_OBSERVATION_TYPES constant
            is a list of valid observation types that can be used to define the expected
            observation type of a wrapper.

            Raises
            ------
            IncompatibleWrapperError
                If the expected observation type is not valid.
            """

            if (not hasattr(self, 'expected_observation_type')
                    or self.expected_observation_type is None):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} needs to have a '
                    f'non None expected observation type '
                    f'defined before applying {observation} decorator.')

            valid = False
            # expected observation type is Space or subset list of [np.ndarray, dict]
            if isinstance(self.expected_observation_type, Space):
                valid = True

            elif isinstance(self.expected_observation_type, list) and set(
                    self.expected_observation_type).issubset(
                        ACCEPTED_OBSERVATION_TYPES):
                valid = True

            if not valid:

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} is defining an exptected observation type '
                    f'of type {type(self.expected_observation_type)}, which is not in the accepted '
                    'observation types {ACCEPTED_OBSERVATION_TYPES}.')

        def _validate_observation_in_expected_observation_type(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> None:
            """
            Validates if the observation is in the expected observation type. the expected observation type
            is a gym.Space or a subset list of ACCEPTED_ACTION_TYPES. If gym.Space, the observation is checked
            the the gym space contains it. If subset list of ACCEPTED_ACTION_TYPES, the observation check
            is less strict and only checks if the observation is of the expected type. This is used when a 
            wrapper can handle an arbitrary numpy array or dict without imposing any additional restrictions
            on strcutre of input.

            ----------
            observation : Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
                The observation to check.

            Raises
            ------
            IncompatibleWrapperError
                If the observation is not in the expected observation type.
            """

            validate_observation(self, observation)

            valid = True

            if isinstance(
                    self.expected_observation_type, Space
            ) and not self.expected_observation_type.contains(observation):
                valid = False

            elif type(observation) not in self.expected_observation_type:
                valid = False

            if not valid:
                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} received an observation of type {type(observation)}, '
                    f'which is not in the expected observation type {self.expected_observation_type}.'
                )

            return None

        def _validate_observation_in_observation_space(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> None:
            """
            Validates if the observation is in the observation space. This is post input validation and performed
            at each time wrapper prouces an observation, to ensure that the observation is in the observation
            space of the wrapper.

            Parameters
            ----------
            observation : Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
                The observation to check.

            Raises
            ------
            IncompatibleWrapperError
                If the observation is not in the observation space.
            """

            validate_observation(self, observation)
            if not self.self.observation_space.contains(observation):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} outputs an observation '
                    f'that is not in its defined observation space {self.exptected_observation_space}.'
                )

            return None

        def observation(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
            """
            This overrides the observation method of the base class and performs the sanity checks
            before and after calling the observation method of the base class. It ensures adherence to
            the input/output structure of the wrapper and catching irregularities in the observation
            during the entire operation of wrapper.


            Parameters
            ----------
            observation : Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
                The observation to check.

            Raises
            ------
            IncompatibleWrapperError
                If the observation is not in the observation space.

            Returns
            -------
            Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
                The result of calling the observation method of the base class.
            """

            self._validate_observation_in_expected_observation_type(observation)
            observation = super().observation(observation)
            self._validate_observation_in_observation_space(observation)

            return observation

    return ObservationSpaceCheckerWrapper


def buffer(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:
    """
    A decorator that augments an existing Gym wrapper class by recursively
    searching through enclosed wrappers for an observation buffer and creating
    a pointer to it, if available. If no observation buffer is found, an error
    is raised. Useful for upstream wrappers that need to access the history of
    observations.

    Args
    ----------
    wrapper_class : type[gym.Wrapper]
        The base Gym wrapper class to be augmented. This should be a subclass of `gym.Wrapper`.

    Returns
    -------
    type[gym.Wrapper]
        A new wrapper class that augments the input wrapper class by creating a pointer to any
        observation buffer found in the enclosed wrappers, if applicable.

    Raises
    ------
    TypeError
        If `wrapper_class` is not a subclass of `gym.Wrapper`.

    Example
    -------

    >>> @buffer
    >>> class CustomBufferDependentWrapper(Wrapper):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__()
    ...     def observation(self, observation):
    ...         buffer = self.observation_buffer_wrapper.observation_buffer
    ...         last_n_observations = buffer[-n:]
    """

    if not issubclass(wrapper_class, Wrapper):
        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class ObservationBufferDependentWrapper(wrapper_class):
        """
        A class that extends the base wrapper class to search recursively through enclosed wrappers for an
        observation buffer and creates a pointer to it. If search fails, an error
        is raised.

        Attributes
        ----------
        observation_buffer_wrapper : ObservationBufferWrapper
            A reference to the observation buffer wrapper found in the
            enclosed wrappers. use this attribute to access the observation
            buffer wrapper and its attributes.

        Methods
        -------
        __init__(self, env: Env, *args, **kwargs) -> None
            Initializes the ObservationBufferDependentWrapper instance.
        find_observation_buffer_wrapper(self, env: Env) -> ObservationBufferWrapper
            Searches recursively for an observation buffer wrapper in
            enclosed wrappers.

        Args
        ----
        env (gym.Env): The environment being wrapped.

        Raises
        ------
        IncompatibleWrapperError: If no observation buffer is found in
            any of the enclosed wrappers.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:
            """
            Initializes the ObservationBufferDependentWrapper instance.

            Args
            ----------
            env : gym.Env
                The environment being wrapped.
            *args : tuple
                Optional arguments to pass to the wrapper.
            **kwargs : dict
                Optional keyword arguments to pass to the wrapper.
            """

            self.observation_buffer_wrapper = self.find_observation_buffer_wrapper(
                env)
            super().__init__(env, *args, **kwargs)

        def find_observation_buffer_wrapper(
                self, env: Env) -> ObservationBufferWrapper:
            """
            Searches recursively for an observation buffer wrapper in enclosed wrappers.

            Args
            ----------
            env : gym.Env
                The environment being wrapped.

            Raises
            ------
            IncompatibleWrapperError
                If no observation buffer is found in any of the enclosed wrappers.

            Returns
            -------
            ObservationBufferWrapper
                The first observation buffer found.
            """

            if isinstance(env, ObservationBufferWrapper):
                return env

            if hasattr(env, 'env'):
                return self.find_observation_buffer_wrapper(env.env)

            else:
                raise IncompatibleWrapperError(
                    f'{wrapper_class.__name__} requires an observation buffer in one of '
                    'the enclosed wrappers.')

    return ObservationBufferDependentWrapper


@observation
@metadata
class PositionsFeatureEngineeringWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that augments observations such that,
    instead of asset quantities held, the notional USD value of assets (positions) is 
    used. This is useful to reflect the value of investments in each asset.

    Attributes
    ----------
    env : Env
        The trading environment to be wrapped.
    n_symbols : int
        The number of assets in the environment.
    n_features : int
        The number of additional features included in each observation after augmentation.

    Methods
    -------
    __init__(self, env: Env) -> None
        Initializes a new instance of the PositionsFeatureEngineeringWrapper class.
    observation(self, observation: Dict[str, np.ndarray[float]]) -> Dict[str, np.ndarray[float]]:
        Augments the observation such that, instead of asset quantities held, the USD value 
        of assets (positions) is used.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import PositionsFeatureEngineeringWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = PositionsFeatureEngineeringWrapper(env)
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes a new instance of the PositionsFeatureEngineeringWrapper class.

        Args
        ----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.n_features = self.market_metadata_wrapper.n_features
        self.expected_observation_type = spaces.Dict({
            'cash':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(1, ),
                       dtype=GLOBAL_DATA_TYPE),
            'asset_quantities':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=GLOBAL_DATA_TYPE),
            'holds':
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=np.int32),
            'features':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_features, ),
                       dtype=GLOBAL_DATA_TYPE)
        })

        self.observation_space = spaces.Dict({
            'cash':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(1, ),
                       dtype=GLOBAL_DATA_TYPE),
            'positions':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=GLOBAL_DATA_TYPE),
            'holds':
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=np.int32),
            'features':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_features, ),
                       dtype=GLOBAL_DATA_TYPE)
        })

        return None

    def observation(
        self,
        observation: Dict[str,
                          np.ndarray[float]]) -> Dict[str, np.ndarray[float]]:
        """
        Augments the observation such that, instead of asset quantities held, the notional 
        value of assets (positions) is used.

        Parameters
        ----------
        observation : dict
            A dictionary containing the original observation.

        Returns
        -------
        dict
            A dictionary containing the augmented observation, where the 'positions' 
            key contains the USD value of each asset.
        """

        asset_prices = self.market_metadata_wrapper.asset_prices
        asset_quantities = observation.pop('asset_quantities')

        observation['positions'] = asset_prices * asset_quantities

        return observation


@observation
@metadata
class WealthAgnosticFeatureEngineeringWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that augments observations
    such that net worth sensitive features are now independent of net worth.
    This wrapper should be applied immediately after the PositionsFeatureEngineeringWrapper.

    Attributes
    ----------
    env : Env
        The trading environment to be wrapped.
    initial_cash : float
        The initial amount of cash in the environment.
    n_symbols : int
        The number of assets in the environment.
    n_features : int
        The number of additional features included in each observation after augmentation.

    Methods
    -------
    observation(observation: Dict[str, np.ndarray[float]]) -> Dict[str, np.ndarray[float]]:
        Augments the observation such that net worth sensitive
        features now have net worth independent values.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import WealthAgnosticFeatureEngineeringWrapper
    >>> from neural.meta.env.wrapper.observation import PositionsFeatureEngineeringWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = PositionsFeatureEngineeringWrapper(env)
    >>> env = WealthAgnosticFeatureEngineeringWrapper(env)
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes a new instance of the WealthAgnosticFeatureEngineeringWrapper class.

        Parameters
        ----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)

        self.initial_cash = self.market_metadata_wrapper.initial_cash
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.n_features = self.market_metadata_wrapper.n_features

        self.expected_observation_type = spaces.Dict({
            'cash':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(1, ),
                       dtype=GLOBAL_DATA_TYPE),
            'positions':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=GLOBAL_DATA_TYPE),
            'holds':
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=np.int32),
            'features':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_features, ),
                       dtype=GLOBAL_DATA_TYPE)
        })

        self.observation_space = spaces.Dict({
            'cash':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(1, ),
                       dtype=GLOBAL_DATA_TYPE),
            'positions':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=GLOBAL_DATA_TYPE),
            'holds':
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.n_symbols, ),
                       dtype=np.int32),
            'features':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.n_features, ),
                       dtype=GLOBAL_DATA_TYPE),
            'return':
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(1, ),
                       dtype=GLOBAL_DATA_TYPE),
        })

        return None

    def observation(
        self,
        observation: Dict[str,
                          np.ndarray[float]]) -> Dict[str, np.ndarray[float]]:
        """
        Augments the observation such that net worth sensitive 
        features now have net worth independent values.

        Parameters
        ----------
        observation : dict
            A dictionary containing the original observation.

        Returns
        -------
        dict
            A dictionary containing the augmented observation, where the 
            'features' key contains net worth sensitive features that are now independent of net worth.
        """

        net_worth = self.market_metadata_wrapper.net_worth

        observation['positions'] /= net_worth
        observation['cash'] /= net_worth
        observation['return'] = net_worth / self.initial_cash - 1

        return observation


class ObservationBufferWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that provides a temporary buffer of observations for subsequent wrappers
    that require this form of information. Autofills itself with the first observation received from the environment to
    maintain a constant buffer size at all times.

    Attributes
    ----------
    env : Env
        The trading environment to be wrapped.
    buffer_size : int
        The maximum number of observations to be stored in the buffer.
    observation_buffer : deque
        A deque object that stores the last n observations, where n is equal to the buffer_size.
        Deque has a self fill property such that when empty it autofills with first input to always
        maintain a fixed size. 

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import WealthAgnosticFeatureEngineeringWrapper
    >>> from neural.meta.env.wrapper.observation import PositionsFeatureEngineeringWrapper
    >>> from nerual.meta.env.wrapper.observation import ObservationBufferWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = PositionsFeatureEngineeringWrapper(env)
    >>> env = WealthAgnosticFeatureEngineeringWrapper(env)
    >>> env = ObservationBufferWrapperf(env)
    """

    def __init__(self, env: Env, buffer_size: int = 10) -> None:
        """
        Initializes a new instance of the ObservationBufferWrapper class.

        Parameters
        ----------
        env : Env
            The trading environment to be wrapped.
        buffer_size : int, optional
            The maximum number of observations to be stored in the buffer. Defaults to 10.
        """

        super().__init__(env)

        assert buffer_size > 0, "The buffer size must be greater than 0."

        self.buffer_size = buffer_size
        self.observation_buffer = FillDeque(buffer_size=buffer_size)

        return None

    def reset(self):
        """
        Resets the environment and clears the observation buffer.

        Returns
        -------
        dict
            A dictionary containing the first observation of the reset environment.
        """

        observation = self.env.reset()
        self.observation_buffer.clear()
        self.observation_buffer.append(observation)

        return observation

    def observation(self, observation):
        """
        Adds the observation to the buffer and returns the buffer as the new observation.

        Parameters
        ----------
        observation : dict
            A dictionary containing the current observation.

        Returns
        -------
        deque
            A deque object containing the last n observations, where n is equal to the buffer_size.
        """

        self.observation_buffer.append(observation)

        return observation


@observation
class FlattenToNUmpyObservationWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that flattens the observation space to a 1D numpy array.

    Attributes
    ----------
    env : Env
        The trading environment to be wrapped.

    Methods
    ----------
    __init__(self, env: Env) -> None:
        Initializes a new instance of the FlattenToNumpyObservationWrapper class.

    flattened_space(self, space: Space) -> spaces.Box:
        Returns a flattened observation space.

    observation(self, observation):
        Flattens the observation space to a 1D numpy array.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import FlattenToNUmpyObservationWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = FlattenToNUmpyObservationWrapper(env)
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes a new instance of the FlattenToNumpyObservationWrapper class.

        Args
        ----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)
        self.expected_observation_type = [dict, np.ndarray]
        self.observation_space = self.flattened_space(env.observation_space)

        return None

    def flattened_space(self, space: Space) -> spaces.Box:
        """
        Returns a flattened observation space.

        Args:
        -----------
        env : Env
            The trading environment.

        Returns:
        --------
        spaces.Box
            The flattened observation space.
        """

        # self.observation_space at constructor is set equal to of self.env.observation_space
        # i.e. observation_spcae of enclosed wrapper due to inheritance from ObservationWrapper
        sample_observation = space.sample()

        if isinstance(sample_observation, np.ndarray):

            flattened_shape = sample_observation.flatten().shape
            flattened_low = space.low.flatten() if not np.isscalar(
                space.low) else np.full(flattened_shape, space.low)
            flattened_high = space.high.flatten() if not np.isscalar(
                space.high) else np.full(flattened_shape, space.high)

            flattened_observation_space = spaces.Box(low=flattened_low,
                                                     high=flattened_high,
                                                     shape=flattened_shape,
                                                     dtype=GLOBAL_DATA_TYPE)

            return flattened_observation_space

        elif isinstance(sample_observation, dict):

            flattened_size = 0
            flattened_observation_space = dict()

            for key, space in space.items():
                flattened_observation_space[key] = self.flattened_space(space)

            flattened_low = np.concatenate([
                flattened_observation_space[key].low
                for key in flattened_observation_space
            ])

            flattened_high = np.concatenate([
                flattened_observation_space[key].high
                for key in flattened_observation_space
            ])

            flattened_size = sum(
                shape[0] for shape in flattened_observation_space.values())

            flattened_shape = (flattened_size, )

            return spaces.Box(low=flattened_low,
                              high=flattened_high,
                              shape=flattened_shape,
                              dtype=GLOBAL_DATA_TYPE)

    def observation(self, observation):
        """
        Flattens the observation space to a 1D numpy array.

        Args:
        -----------
        observation : dict or ndarray
            The observation space.

        Returns:
        --------
        ndarray
            The flattened observation space.
        """

        if isinstance(observation, dict):
            observation = np.concatenate(
                [array.flatten() for array in observation.values()])

        elif isinstance(observation, np.ndarray):
            observation = observation.flatten()

        return observation


@buffer
class ObservationStackerWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that stacks the last n observations in the buffer.
    If observation is changed between buffer and stacker, all changes will be lost as this wrapper's
    point of reference is the buffer.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    stack_size : int
        The number of observations to be concatenated.

    Methods:
    --------
    __init__(self, env: Env, stack_size: Optional[int] = None) -> None:
        Initializes a new instance of the ObservationStackerWrapper class.

    infer_stacked_observation_space(self, observation_space: spaces.Box) -> spaces.Box:
        Infers the observation space of the stacked observations.

    stacked_observation_space(self) -> Space:
        Returns the observation space of the stacked observations.

    observation(self, observation: Dict[str, np.ndarray[float]] | np.ndarray[float]):
        Returns the last n stacked observations in the buffer. Note observation in argument
        is discarded and only elemetns in buffer are used, thus if observation is changed
        between buffer and stacker, all changes will be lost as this wrapper's point of
        reference is the buffer.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import FlattenToNUmpyObservationWrapper
    >>> from neural.meta.env.wrapper.observation import ObservationStackerWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = FlattenToNUmpyObservationWrapper(env)
    >>> env = ObservationStackerWrapper(env)
    """

    def __init__(self, env: Env, stack_size: Optional[int] = None) -> None:
        """
        Initializes a new instance of the ObservationStackerWrapper class.

        Parameters:
        -----------
        env : Env
            The trading environment to be wrapped.
        stack_size : int, optional
            The number of observations to be concatenated. Defaults to 4.
        """

        super().__init__(env)

        buffer_size = self.observation_buffer_wrapper.buffer_size
        self.stack_size = (stack_size
                           if stack_size is not None else buffer_size)

        assert (
            self.stack_size <= buffer_size,
            f"Stack size {self.stack_size} cannot exceed buffer size {buffer_size}."
        )

        self.observation_space = None

        return None

    def infer_stacked_observation_space(
            self, observation_space: spaces.Box) -> spaces.Box:
        """
        Infers the observation space of the stacked observations.

        Parameters:
        -----------
        observation_space : spaces.Box
            The original observation space.

        Returns:
        --------
        spaces.Box
            The observation space of the stacked observations.
        """

        # TODO: Add support for other observation spaces
        assert isinstance(observation_space, spaces.Box), (
            f"currently {ObservationStackerWrapper.__name__} only supports Box observation spaces"
        )

        observation = observation_space.sample()
        stacked_shape = np.stack([observation] * self.stack_size).shape

        stacked_low = (np.full(stacked_shape, observation_space.low)
                       if np.isscalar(observation_space.low) else np.stack(
                           self.stack_size * [observation_space.low]))

        stacked_high = (np.full(stacked_shape, observation_space.high)
                        if np.isscalar(observation_space.high) else np.stack(
                            self.stack_size * [observation_space.high]))

        return spaces.Box(low=stacked_low,
                          high=stacked_high,
                          shape=stacked_shape,
                          dtype=GLOBAL_DATA_TYPE)

    def stacked_observation_space(self) -> Space:
        """
        Returns a flattened observation space.

        Args:
        -----------
        env : Env
            The trading environment.

        Returns:
        --------
        spaces.Box
            The flattened observation space.
        """

        # self.observation_space at constructor is set equal to of self.env.observation_space
        # i.e. observation_spcae of enclosed wrapper due to inheritance from ObservationWrapper

        buffer_observation_space = self.observation_buffer_wrapper.observation_space

        if isinstance(buffer_observation_space, spaces.Box):
            stacked_observation_space = self.infer_stacked_observation_space(
                buffer_observation_space)

        elif isinstance(buffer_observation_space, dict):
            stacked_observation_space = dict()
            for key, space in buffer_observation_space.items():
                stacked_observation_space[
                    key] = self.infer_stacked_observation_space(space)

        return stacked_observation_space

    def observation(self, observation: Dict[str, np.ndarray[float]]
                    | np.ndarray[float]):
        """
        Returns the last n stacked observations in the buffer. Note observation in argument
        is discarded and only elemetns in buffer are used, thus if observation is changed
        between buffer and stacker, all changes will be lost as this wrapper's point of
        reference is the buffer.

        Parameters:
        -----------
        observation : dict or ndarray
            A dictionary or ndarray containing the current observation.

        Returns:
        --------
        ndarray or dict of ndarrays
            An ndarray or dict of ndarrays containing the stacked observations.
        """

        stack = self.observation_buffer_wrapper.observation_buffer[-self.
                                                                   stack_size:]

        # Check if the observations are ndarrays or dictionaries of ndarrays
        if isinstance(stack[0], np.ndarray):
            stacked_observation = np.stack(stack)  # default axis=0

        elif isinstance(stack[0], dict):
            stacked_observation = {}
            for key in stack[0].keys():
                key_stack = [observation[key] for observation in stack]
                key_stack = np.stack(key_stack)
                stacked_observation[key] = key_stack

        return stacked_observation


@observation
class RunningStatisticsObservationWrapper(ObservationWrapper):
    """
    A Gym environment wrapper that tracks the running mean and standard deviation
    of the observations using the RunningMeanStandardDeviation class.

    Parameters:
    -----------
    env : gym.Env
        The environment to wrap.

    Methods:
    --------
    initialize_observation_rms(observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> RunningMeanStandardDeviation | Dict[str, RunningMeanStandardDeviation]:
        Initializes the running mean and standard deviation for the observations.

    update(observation: np.ndarray[float] | Dict[str, np.ndarray[float]]) -> None:
        Updates the running mean and standard deviation for the observations.

    observation(observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        Returns the observation and updates the running mean and standard deviation.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import FlattenToNUmpyObservationWrapper
    >>> from neural.meta.env.wrapper.observation import ObservationStackerWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = FlattenToNUmpyObservationWrapper(env)
    >>> env = ObservationStackerWrapper(env)
    >>> env = RunningMeanSandardDeviationObservationWrapper(env)
    """

    def __init__(self,
                 env: Env,
                 observation_statistics: Optional[RunningStatistics] = None,
                 track_statistics: bool = True):

        super().__init__(env)

        self.expected_observation_type = [dict, np.ndarray]
        self.observation_statistics = (observation_statistics if
                                       observation_statistics is not None else
                                       self.initialize_observation_statistics(
                                           env.observation_space.sample()))

        if track_statistics:
            observation_statistics = self.observation_statistics

        return self.observation_statistics

    def initialize_observation_statistics(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> RunningStatistics | Dict[str, RunningStatistics]:
        """
        Initializes the running mean standard deviation for the observation.

        Args:
        observation (np.ndarray[float] or Dict[str, np.ndarray[float]]): The observation for which
        the running mean standard deviation needs to be initialized.

        Returns:
        A RunningMeanStandardDeviation object or a dictionary containing RunningMeanStandardDeviation
        objects for each observation key.
        """

        if isinstance(observation, np.ndarray):
            observation_rms = RunningStatistics()

        elif isinstance(observation, dict):

            observation_rms = dict()
            for key in observation.keys():
                observation_rms[key] = RunningStatistics()

        return observation_rms

    def update(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> None:
        """
        Updates the running mean standard deviation with the new observation.

        Args:
        observation (np.ndarray[float] or Dict[str, np.ndarray[float]]): The new observation.

        Returns:
        None.
        """

        if isinstance(observation, np.ndarray):
            self.observation_statistics.update(observation)

        elif isinstance(observation, dict):
            for key, array in observation.items():
                self.observation_statistics[key].update(array)

        return None

    def observation(self, observation: np.ndarray[float]
                    | Dict[str, np.ndarray[float]]):

        self.update(observation)

        return observation


@observation
class ObservationNormalizerWrapper(RunningStatisticsObservationWrapper):
    """
    A wrapper class that normalizes the observations received from the environment
    using running mean standard deviation.

    Args:
    env (gym.Env): The environment to wrap.

    Attributes:
    expected_observation_type (list): A list of expected observation types.
    observation_rms (RunningMeanStandardDeviation or dict): Running mean standard deviation for the observation.

    Methods:
    initialize_observation_rms(observation):
        Initializes the running mean standard deviation for the observation.
    update(observation):
        Updates the running mean standard deviation with the new observation.
    observation(observation):
        Normalizes the observation received from the environment and returns it.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import NormalizeObservationWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = NormalizeObservationWrapper(env)
    """

    def __init__(self,
                 env: Env,
                 epsilon: float = 1e-8,
                 clip: float = 10,
                 observation_statistics=Optional[RunningStatistics],
                ) -> None:

        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self. = track_statistics

    def observation(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Normalizes the observation received from the environment and returns it.

        Args:
            observation (np.ndarray[float] or Dict[str, np.ndarray[float]]): The observation to normalize.

        Returns:
            The normalized observation.
        """

        observation = super().observation(observation)

        if isinstance(observation, np.ndarray):
            observation = self.observation_rms.normalize(
                observation, self.epsilon, self.clip).astype(GLOBAL_DATA_TYPE)

        elif isinstance(observation, dict):

            observation = dict()

            for key, rms in self.observation_rms.items():
                observation[key] = rms.normalize(
                    observation, self.epsilon,
                    self.clip).astype(GLOBAL_DATA_TYPE)

        return observation


@buffer
class FinancialIndicatorsWrapper(ObservationWrapper):

    # computes running financial indicators such as CCI, MACD
    # etc. Requires an observations buffer containing a window
    # of consecutive observations.

    def observation(self, observation):
        return None
