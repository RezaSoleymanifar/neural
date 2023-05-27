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
    This is a helper function that is shared between the observation
    type checkers. It performs a basic check to see if the observation
    is a known observation type accepted by this library. Aceepted
    observation types are:
        - np.ndarray
        - Dict[str, np.ndarray] (all values must be np.ndarray)

    Args
    ----------
    wrapper : Wrapper
        The wrapper object that receives the observation. Used to
        produce hint when raising an error.
    observation : np.ndarray or Dict[str, np.ndarray]
        The observation to be validated. If it is a dictionary, all its
        values must be numpy arrays.

    Raises
    ------
    IncompatibleWrapperError
        If the wrapper received an observation of an incompatible type.
    """
    if isinstance(observation, dict):
        if all(isinstance(observation[key], np.ndarray) for key in observation):
            valid = True

    elif isinstance(observation, np.ndarray):
        valid = True

    if not valid:
        raise IncompatibleWrapperError(
            f'Wrapper {type(wrapper).__name__} received an observation of '
            f'type {type(observation)}, which is not in the accepted '
            f'observation types {ACCEPTED_OBSERVATION_TYPES}.')

    return None


def observation(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:
    """
    A decorator that augments an existing Gym wrapper and extends it to
    have input and output self checks. The result is a new observation
    wrapper that checks if the observation of the wrapped env is in its
    expected observation space, and also checks if the observation
    returned by the wrapped env is in its defined observation space.

    Args
    ----------
    wrapper_class : Type[Wrapper]
        The base gym wrapper class to be augmented. This should be a
        subclass of `gym.Wrapper`.

    Raises
    ------
    TypeError
        If the `wrapper_class` argument is not a subclass of
        `gym.Wrapper`.
    IncompatibleWrapperError
        If the observation space is not defined in the enclosed
        environment, or if the expected observation type is not valid.

    Returns
    -------
    Type[Wrapper]
        A new wrapper class that checks if an observation is in the
        observation space before returning it from the reset and step
        functions.

    Examples
    --------
    In this example the observation decorator is used to:
        - Check if the env at constructor has an observation space of 
            type gym.spaces.
        - Check if the observation returned by the wrapped env is in
            spaces.Box(low=0, high=1, shape=(1,)), e.g. array =
            np.array([0.5]) is valid, but array = np.array([1.5]) is
            not.
    >>> from gym import Wrapper
    >>> from neural.meta.env.wrapper.observation import observation
    >>> @observation
    ... class CustomObservationWrapper(ObservationWrapper):
    ...     def __init__(self, env):
    ...     self.expected_observation_type = [np.ndarray]
    ...     self.observation_space = spaces.Box(low=0, high=1, shape=(1,))
    ...     def observation(self, observation):
    ...         pass

    Notes
    -----
    Read more about how observation wrappers work from
    gym.ObservationWrapper
    """

    if not issubclass(wrapper_class, ObservationWrapper):
        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {ObservationWrapper}"
        )

    class ObservationSpaceCheckerWrapper(wrapper_class):
        """
        This wrapper allows an observation wrapper to have input and
        output self checks by subclassing from it and overriding the
        observation method. The result is a new observation wrapper that
        checks if the observation of the wrapped env is in its expected
        observation space, and also check if the observation produced by
        the wrapped env is in its defined observation space.

        Args
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
        _validate_expected_observation_type(self) -> None:
            Validates the expected observation type of the wrapper.
            Makes sure that the expected observation type is a list of
            valid types. Valid types are:
                - np.ndarray
                - Dict[str, np.ndarray] (all values must be np.ndarray) 
                - gym.spaces.Space
        _validate_observation_in_expected_observation_type(self, 
            observation: Union[np.ndarray[float], Dict[str,
            np.ndarray[float]]] ) -> None: 
                Validates if the observation is in the expected
                observation type.
        _validate_observation_in_observation_space(self,
            observation: Union[np.ndarray[float], Dict[str,
            np.ndarray[float]]]) -> None: Validates if the observation
            is in the observation space.
        observation(self, observation: Union[np.ndarray[float],
        Dict[str, np.ndarray[float]]]
            ) -> Union[np.ndarray[float], Dict[str, np.ndarray[float]]]:
            Performs both input/output observation checks and returns
            the observation.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:
            """
            Initializes the ObservationSpaceCheckerWrapper instance.

            Args
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

            Notes:
            ------
            Before calling the super constructor, this 
            """

            if not hasattr(env, 'observation_space') or not isinstance(
                    env.observation_space, Space):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} requires a non None '
                    f'observation space of type {Space} to be defined in '
                    f'the enclosed environment {type(env).__name__}.')

            super().__init__(env, *args, **kwargs)

            self._validate_expected_observation_type()
            self._validate_observation_in_expected_observation_type(
                env.observation_space.sample())

            return None

        def _validate_expected_observation_type(self) -> None:
            """
            Expected observation type is also the type of the
            observation that the enclosed environment should return.
            Validates the expected observation type of the wrapper.
            ACCEPTED_OBSERVATION_TYPES constant is a list of valid
            observation types that can be used to define the expected
            observation type of a wrapper. Expected observation type is
            Space or subset list of [np.ndarray, dict].

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

            if isinstance(self.expected_observation_type, Space):
                valid = True

            elif isinstance(self.expected_observation_type, list) and set(
                    self.expected_observation_type).issubset(
                        ACCEPTED_OBSERVATION_TYPES):
                valid = True

            if not valid:

                raise IncompatibleWrapperError(
                    f'Wrapper {type(super()).__name__} is defining an exptected '
                    f'observation type of type {type(self.expected_observation_type)} '
                    f', which is not in the accepted observation types '
                    f' {ACCEPTED_OBSERVATION_TYPES}.')

        def _validate_observation_in_expected_observation_type(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> None:
            """
            Validates if the observation is in the expected observation
            type. the expected observation type is a gym.Space or a
            subset list of ACCEPTED_ACTION_TYPES. If gym.Space, the
            observation is checked the the gym space contains it. If
            subset list of ACCEPTED_ACTION_TYPES, the observation check
            is less strict and only checks if the observation is of the
            expected type. This is used when a wrapper can handle an
            arbitrary numpy array or dict without imposing any
            additional restrictions on strcutre of input.

            ----------
            observation : Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
                The observation to check.

            Raises
            ------
            IncompatibleWrapperError
                If the observation is not in the expected observation
                type.
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
                    f'Wrapper {type(self).__name__} received an observation '
                    f'of type {type(observation)}, which is not in the '
                    f'expected observation type {self.expected_observation_type}.'
                )

            return None

        def _validate_observation_in_observation_space(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> None:
            """
            Validates if the observation is in the observation space.
            This is post input validation and performed at each time
            wrapper prouces an observation, to ensure that the
            observation is in the observation space of the wrapper.

            Args
            ----------
            observation : Union[np.ndarray[float], Dict[str,
            np.ndarray[float]]]
                The observation to check.

            Raises
            ------
            IncompatibleWrapperError
                If the observation is not in the observation space.
            """
            validate_observation(self, observation)
            if not self.self.observation_space.contains(observation):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(super()).__name__} outputs an observation '
                    f'that is not in its defined observation space '
                    f'{self.observation_space}.')

            return None

        def observation(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
            """
            This overrides the observation method of the base class and
            performs the sanity checks before and after calling the
            observation method of the base class. It ensures adherence
            to the input/output structure of the wrapper and catching
            irregularities in the observation during the entire
            operation of wrapper.


            Args
            ----------
            observation : Union[np.ndarray[float], Dict[str,
            np.ndarray[float]]]
                The observation to check.

            Returns
            -------
            Union[np.ndarray[float], Dict[str, np.ndarray[float]]]
                The result of calling the observation method of the base
                class.
            """

            self._validate_observation_in_expected_observation_type(observation)
            observation = super().observation(observation)
            self._validate_observation_in_observation_space(observation)

            return observation

    return ObservationSpaceCheckerWrapper


def buffer(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:
    """
    A decorator that augments an existing Gym wrapper class by
    recursively searching through enclosed wrappers for an observation
    buffer and creating a pointer to it, if available. If no observation
    buffer is found, an error is raised. Useful for upstream wrappers
    that need to access the history of observations.

    Args
    ----------
    wrapper_class : type[gym.Wrapper]
        The base Gym wrapper class to be augmented. This should be a
        subclass of `gym.Wrapper`.

    Returns
    -------
    type[gym.Wrapper]
        A new wrapper class that augments the input wrapper class by
        creating a pointer to the first observation buffer found in the
        enclosed wrappers, if applicable.

    Raises
    ------
    TypeError
        If `wrapper_class` is not a subclass of `gym.Wrapper`.

    Example
    -------
    >>> from neural.wrapper.observation import buffer
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
        A class that extends the base wrapper class to search
        recursively through enclosed wrappers for an observation buffer
        and creates a pointer to it. If search fails, an error is
        raised.

        Args
        ----
            env (gym.Env): The environment being wrapped.

        Attributes
        ----------
            observation_buffer_wrapper : ObservationBufferWrapper
                A reference to the observation buffer wrapper found in the
                enclosed wrappers. use this attribute to access the
                observation buffer wrapper and its attributes.

        Methods
        -------
            find_observation_buffer_wrapper(self, env: Env) ->
            ObservationBufferWrapper
                Searches recursively for an observation buffer wrapper in
                enclosed wrappers.

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
            Searches recursively for an observation buffer wrapper in
            enclosed wrappers.

            Args
            ----------
                env : gym.Env
                    The environment being wrapped.

            Raises
            ------
                IncompatibleWrapperError
                    If no observation buffer is found in any of the enclosed
                    wrappers.

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
                    f'{wrapper_class.__name__} requires an observation '
                    'buffer in one of the enclosed wrappers.')

    return ObservationBufferDependentWrapper


@observation
@metadata
class PositionsFeatureEngineeringWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that augments
    observations such that, instead of asset quantities held, the
    notional USD value of assets (positions) is used. This is useful to
    reflect the value of investments in each asset, instead of reporting
    the quantity of assets held.

    Attributes
    ----------
        env : Env
            The trading environment to be wrapped.
        n_assets : int
            The number of assets in the environment.
        n_features : int
            The number of additional features included in each observation
            after augmentation.
        expected_observation_type : spaces.Dict
            The expected observation space of the wrapped environment.
        observation_space : spaces.Dict
            The observation space of the wrapped environment.

    Methods
    -------
        observation(self, observation: Dict[str, np.ndarray[float]]) ->
        Dict[str, np.ndarray[float]]:
            Augments the observation such that, instead of asset quantities
            held, the notional value of assets (positions) is used. If asset is
            borrowed (negative quantity), the notional value is negative.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import PositionsFeatureEngineeringWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = PositionsFeatureEngineeringWrapper(env)
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes a new instance of the PositionsFeatureEngineeringWrapper
        class.

        Args
        ----------
        env : Env
            The trading environment to be wrapped.
        """
        super().__init__(env)
        self.n_assets = self.market_metadata_wrapper.n_assets
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
        Augments the observation such that, instead of asset quantities held,
        the notional value of assets (positions) is used.

        Parameters
        ----------
            observation : Dict[str, np.ndarray[float]]
                A dictionary containing the original observation.

        Returns
        -------
            observation : Dict[str, np.ndarray[float]]
                A dictionary containing the augmented observation, where the
                'positions' key contains the USD value of each asset.
        """

        asset_prices = self.market_metadata_wrapper.asset_prices
        asset_quantities = observation.pop('asset_quantities')

        observation['positions'] = asset_prices * asset_quantities

        return observation


@observation
@metadata
class WealthAgnosticFeatureEngineeringWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that augments
    observations such that net worth sensitive features are now
    independent of net worth. This wrapper should be applied immediately
    after the PositionsFeatureEngineeringWrapper.

    Attributes
    ----------
        env : Env
            The trading environment to be wrapped.
        initial_cash : float
            The initial amount of cash in the environment.
        n_symbols : int
            The number of assets in the environment.
        n_features : int
            The number of additional features included in each observation
            after augmentation.

    Methods
    -------
        observation(observation: Dict[str, np.ndarray[float]]) -> Dict[str,
        np.ndarray[float]]:
            Augments the observation such that net worth sensitive features
            now have net worth independent values.

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
        Initializes a new instance of the
        WealthAgnosticFeatureEngineeringWrapper class.

        Args
        ----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)

        self.initial_cash = self.market_metadata_wrapper.initial_cash
        self.n_assets = self.market_metadata_wrapper.n_assets
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
                       shape=(self.n_assets, ),
                       dtype=GLOBAL_DATA_TYPE),
            'holds':
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.n_assets, ),
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
                       shape=(self.n_assets, ),
                       dtype=GLOBAL_DATA_TYPE),
            'holds':
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.n_assets, ),
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
        Augments the observation such that wealth dependent features
        now have wealth normalized values.

        Args
        ----------
            observation : Dict[str, np.ndarray[float]]
                A dictionary containing the original observation.

        Returns
        -------
            observation: Dict[str, np.ndarray[float]]
                A dictionary containing the augmented observation. Adds
        """

        equity = self.market_metadata_wrapper.equity

        observation['positions'] /= equity
        observation['cash'] /= equity

        return observation


class ObservationBufferWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that provides a temporary
    buffer of observations for subsequent wrappers that require this form of
    information. Autofills itself with the first observation received from the
    environment to maintain a constant buffer size at all times.

    Attributes
    ----------
        env : Env
            The trading environment to be wrapped.
        buffer_size : int
            The maximum number of observations to be stored in the buffer.
        observation_buffer : deque
            A deque object that stores the last n observations, where n is
            equal to the buffer_size. Deque has a self fill property such that
            when empty it autofills with first input to always maintain a fixed
            size. 

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from nerual.meta.env.wrapper.observation import ObservationBufferWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = ObservationBufferWrapper(env)
    """

    def __init__(self, env: Env, buffer_size: int = 10) -> None:
        """
        Initializes a new instance of the ObservationBufferWrapper
        class.

        Args
        ----------
            env : Env
                The trading environment to be wrapped.
            buffer_size : int, optional
                The maximum number of observations to be stored in the
                buffer. Defaults to 10.
        """

        super().__init__(env)

        if not isinstance(buffer_size, int) or not buffer_size > 0:
            raise ValueError("The buffer size must be positive integer.")

        self.buffer_size = buffer_size
        self.observation_buffer = FillDeque(buffer_size=buffer_size)

        return None

    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Resets the environment and clears the observation buffer.

        Returns
        -------
        Dict[str, np.ndarray[float]]
            A dictionary containing the first observation of the reset
            environment.
        """

        observation = self.env.reset()
        self.observation_buffer.clear()
        self.observation_buffer.append(observation)

        return observation

    def observation(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Adds the observation to the buffer and returns the observation
        without modification.

        Args
        ----------
            observation : np.ndarray[float] | Dict[str, np.ndarray[float]]  
                The observation to be added to the buffer.


        Returns
        -------
            observation : np.ndarray[float] | Dict[str, np.ndarray[float]]
                The observation without modification.
        """
        self.observation_buffer.append(observation)

        return observation


@observation
class FlattenToNUmpyObservationWrapper(ObservationWrapper):
    """
    A wrapper for OpenAI Gym trading environments that flattens the observation
    space to a 1D numpy array. If numpy array then array is flattened. If
    dictionary then each value is flattened and concatenated together to
    produce a 1D numpy array.

    Attributes
    ----------
        env : Env
            The trading environment to be wrapped.

    Methods
    ----------
        flattened_space: Space -> spaces.Box
            Returns a flattened observation space.
        observation: np.ndarray[float] | Dict[str, np.ndarray[float]] ->
        np.ndarray[float] | Dict[str, np.ndarray[float]]
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
        Initializes a new instance of the
        FlattenToNumpyObservationWrapper class.

        Args
        ----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)
        self.expected_observation_type = [dict, np.ndarray]
        self.observation_space = self.flattened_space(self.observation_space)

        return None

    def flattened_space(self, space: Space) -> spaces.Box:
        """
        Returns a flattened observation space. self.observation_space at
        constructor is set equal to that of self.env.observation_space i.e.
        observation_space of enclosed wrapper due to inheritance from
        ObservationWrapper. This method is used to set the observation_space
        attribute to a flattened version of the observation space of the
        enclosed wrapper. Having defined the observation_space attribute
        is necessary for some RL training algorithms to work.

        Args:
        -----------
            env : Env
                The trading environment.

        Returns:
        --------
            spaces.Box
                The flattened observation space.
        """

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
                space.shape[0]
                for space in flattened_observation_space.values())

            flattened_shape = (flattened_size, )

            return spaces.Box(low=flattened_low,
                              high=flattened_high,
                              shape=flattened_shape,
                              dtype=GLOBAL_DATA_TYPE)

    def observation(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Flattens the observation space to a 1D numpy array. This is separate
        from flattening the observation space in the constructor.

        Args:
        -----------
            observation : np.ndarray[float] | Dict[str, np.ndarray[float]]
            -> np.ndarray[float] | Dict[str, np.ndarray[float]]
                The observation to be flattened.

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
    A wrapper for OpenAI Gym trading environments that stacks the last n
    observations in the buffer. If observation is changed between buffer
    and stacker, all changes will be lost as this wrapper's point of
    reference is the buffer.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    stack_size : int
        The number of observations to be concatenated.

    Methods:
    --------
        infer_stacked_observation_space: spaces.Box -> spaces.Box
            Returns the observation space of the stacked observations.

        stacked_observation_space: -> spaces.Box | spaces.Dict

    observation: Dict[str, np.ndarray[float]] | np.ndarray[float]) ->
    Dict[str, np.ndarray[float]] | np.ndarray[float]:
        Returns the last n stacked observations in the buffer. Note
        observation in argument is discarded and only elements in buffer
        are used, thus if observation is changed between buffer and
        stacker, all changes will be lost as this wrapper's point of
        reference is the buffer.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import ObservationStackerWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = ObservationStackerWrapper(env)
    """

    def __init__(self, env: Env, stack_size: Optional[int] = None) -> None:
        """
        Initializes a new instance of the ObservationStackerWrapper
        class.

        Args:
        -----------
        env : Env
            The trading environment to be wrapped.
        stack_size : int, optional
            The number of observations to be concatenated. Defaults to
            4.
        """

        super().__init__(env)

        buffer_size = self.observation_buffer_wrapper.buffer_size
        self.stack_size = (stack_size
                           if stack_size is not None else buffer_size)

        if stack_size > buffer_size:
            raise AssertionError(f'Stack size {stack_size} cannot exceed '
                                 f'buffer size {buffer_size}.')

        self.observation_space = self.stacked_observation_space()

        return None

    def infer_stacked_observation_space(
            self, observation_space: spaces.Box) -> spaces.Box:
        """
        Infers the observation space of the stacked observations. Takes 
        a box observation space and returns a box observation space with
        the shape of the stacked observations.

        Args:
        -----------
            observation_space : spaces.Box
                The original observation space.

        Returns:
        --------
            spaces.Box
                The observation space of the stacked observations.
        
        Raises:
        -------
            AssertionError
                If observation space is not a box space.

        TODO: Add support for other observation spaces.
        """

        if not isinstance(observation_space, spaces.Box):
            raise AssertionError(
                f'currently {ObservationStackerWrapper.__name__} only '
                'supports Box observation spaces')

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

    def stacked_observation_space(self) -> spaces.Box | spaces.Dict:
        """
        The observation space of the stacked observations. Works with
        both spaces.Box and spaces.Dict observation spaces. If the
        observation space is a dict, the stacked observation space will
        be a dict with the same keys and stacked shape of each key. If
        the observation space is a box, the stacked observation space
        will be a box with the shape of the stacked observations.
        Currently only supports spaces.Box observation spaces.

        Returns:
        --------
            Spaces.Box | Spaces.Dict
                The observation space of the stacked observations.
        """
        buffer_observation_space = (
            self.observation_buffer_wrapper.observation_space)

        if isinstance(buffer_observation_space, spaces.Box):
            stacked_observation_space = self.infer_stacked_observation_space(
                buffer_observation_space)

        elif isinstance(buffer_observation_space, dict):
            stacked_observation_space = dict()
            for key, space in buffer_observation_space.items():
                stacked_observation_space[
                    key] = self.infer_stacked_observation_space(space)

        return stacked_observation_space

    def observation(
        self, observation: Dict[str, np.ndarray[float]]
        | np.ndarray[float]
    ) -> Dict[str, np.ndarray[float]] | np.ndarray[float]:
        """
        Returns the last n stacked observations in the buffer. Note
        observation in argument is discarded and only elemetns in buffer
        are used, thus if observation is changed between buffer and
        stacker, all changes will be lost as this wrapper's point of
        reference is the buffer.

        Args:
        -----------
            observation : Dict[str, np.ndarray[float]] |
            np.ndarray[float] -> Dict[str, np.ndarray[float]] |
            np.ndarray[float]
                A dictionary or numpy array containing the current
                observation. Is not used. Only elements in buffer are
                used. If no modifications are made to the observation
                the last item in the buffer will be equal to the
                observation.

        Returns:
        --------
            Dict[str, np.ndarray[float]] | np.ndarray[float]
                The last n stacked observations in the buffer.
        """

        stack = self.observation_buffer_wrapper.observation_buffer[-self.
                                                                   stack_size:]

        if isinstance(stack[0], np.ndarray):
            stacked_observation = np.stack(stack)

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
    A Gym environment wrapper that tracks the running mean and standard
    deviation of the observations using the RunningMeanStandardDeviation
    class.

    Args:
    -----------
    env : gym.Env
        The environment to wrap.

    Methods:
    --------
        observation(observation: np.ndarray[float] | Dict[str,
        np.ndarray[float]] -> np.ndarray[float] | Dict[str,
        np.ndarray[float]])
            Returns the running mean and standard deviation normalized
            observation.
        

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> env = TrainMarketEnv(...)
    >>> env = RunningMeanSandardDeviationObservationWrapper(env)
    """

    def __init__(self,
                 env: Env,
                 epsilon: float = 1e-8,
                 clip_threshold: float = 10,
                 observation_statistics: Optional[RunningStatistics] = None):

        super().__init__(env)

        self.expected_observation_type = [dict, np.ndarray]
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        if observation_statistics is None:
            self.observation_statistics = (
                observation_statistics if observation_statistics is not None
                else self.initialize_observation_statistics(
                    env.observation_space.sample()))
        observation_statistics = self.observation_statistics

        return self.observation_statistics

    def initialize_observation_statistics(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> RunningStatistics | Dict[str, RunningStatistics]:
        """
        Initializes the running mean standard deviation for the
        observation. If the observation is a dictionary, a dictionary
        containing the running mean standard deviation for each key is
        produced.

        Args:
        -----------
            observation (np.ndarray[float] or Dict[str,
            np.ndarray[float]]):
                The observation to initialize the running mean standard
                deviation with. The shape of the observation is used to
                initialize the running mean standard deviation.

        Returns:
        --------
            RunningStatistics | Dict[str, RunningStatistics]
                The running mean standard deviation for the observation.

        Raises:
        -------
            TypeError:
                If the observation is not a numpy array or a dictionary
                containing numpy arrays.
        """

        if isinstance(observation, np.ndarray):
            observation_rms = RunningStatistics(
                epsilon=self.epsilon, clip_threshold=self.clip_threshold)

        elif isinstance(observation, dict):

            observation_rms = dict()
            for key in observation.keys():
                observation_rms[key] = RunningStatistics(
                    epsilon=self.epsilon, clip_threshold=self.clip_threshold)

        return observation_rms

    def update(
            self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> None:
        """
        Updates the running mean standard deviation with the new
        observation.

        Args:
        -----------
            observation (np.ndarray[float] or Dict[str,
            np.ndarray[float]]):
                The observation to update the running mean standard
                deviation with.
        """

        if isinstance(observation, np.ndarray):
            self.observation_statistics.update(observation)

        elif isinstance(observation, dict):
            for key, array in observation.items():
                self.observation_statistics[key].update(array)

        return None

    def observation(
        self, observation: np.ndarray[float]
        | Dict[str, np.ndarray[float]]
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Updates the running mean standard deviation with the new
        observation. If the observation is a dictionary, the running
        mean standard deviation for each key is updated.

        Args:
        -----------
            observation (np.ndarray[float] or Dict[str,
            np.ndarray[float]]):
                The observation to update the running mean standard
                deviation with.
        
        
        """

        self.update(observation)

        return observation


@observation
class ObservationNormalizerWrapper(RunningStatisticsObservationWrapper):
    """
    A Gym environment wrapper that normalizes the observations using the
    RunningMeanStandardDeviation class. If numpy arrays are used, the
    observations are normalized using RunningMeanStandardDeviation. If
    dictionaries are used, each key is normalized using
    RunningMeanStandardDeviation. The observation_statistics argument is
    set equal to the running mean standard deviation of the observation.
    If used in a pipe, argument can be saved as an attribute of the
    pipe, and used to restore the running mean standard deviation of the
    warpper, for example to resume training, or start trading.

    Args:
    -----------
        env : gym.Env
            The environment to wrap.
        observation_statistics : RunningStatistics | Dict[str,
        RunningStatistics] | None
            The running mean standard deviation of the observation. If
            None, the running mean standard deviation is initialized
            using the observation space of the environment. Wrapper sets
            the argument equal to the running mean standard deviation of
            the observation.

    Methods:
    --------
        observation(observation: np.ndarray[float] | Dict[str,
        np.ndarray[float]]) -> np.ndarray[float] | Dict[str,    
        np.ndarray[float]]
            Normalizes the observation received from the environment and


    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.observation import NormalizeObservationWrapper
    >>> env = TrainMarketEnv(...)
    >>> env = NormalizeObservationWrapper(env)
    """

    def __init__(
        self,
        env: Env,
        epsilon: float = 1e-8,
        clip_threshold: float = 10,
        observation_statistics: Optional[RunningStatistics] = None,
    ) -> None:

        super().__init__(env=env,
                         epsilon=epsilon,
                         clip_threshold=clip_threshold,
                         observation_statistics=observation_statistics)
        self.epsilon = epsilon
        self.clip = clip_threshold

    def observation(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Normalizes the observation received from the environment and
        returns it. The superclass updates the running mean standard
        deviation with the new observation.

        Args:
        -----------
            observation (np.ndarray[float] or Dict[str,
            np.ndarray[float]]): The observation to normalize.

        Returns: 
        --------
            The normalized observation. np.ndarray[float] or Dict[str,
            np.ndarray[float]
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

    # computes running financial indicators such as CCI, MACD etc.
    # Requires an observations buffer containing a window of consecutive
    # observations.

    def observation(self, observation):
        return None
