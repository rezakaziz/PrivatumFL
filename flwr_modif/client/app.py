# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client app."""


import argparse
import sys
import time
from logging import INFO, WARN
from pathlib import Path
from typing import Callable, ContextManager, Optional, Tuple, Union

from flwr_modif.client.client import Client
from flwr_modif.client.flower import Flower
from flwr_modif.client.typing import Bwd, ClientFn, Fwd
from flwr_modif.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr_modif.common.address import parse_address
from flwr_modif.common.constant import (
    MISSING_EXTRA_REST,
    TRANSPORT_TYPE_GRPC_BIDI,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPES,
)
from flwr_modif.common.logger import log, warn_experimental_feature
from flwr_modif.proto.task_pb2 import TaskIns, TaskRes

from flwr_modif.client.flower import load_callable
from flwr_modif.client.grpc_client.connection import grpc_connection
from flwr_modif.client.grpc_rere_client.connection import grpc_request_response
from flwr_modif.client.message_handler.message_handler import handle_control_message
from flwr_modif.client.numpy_client import NumPyClient
from flwr_modif.client.workload_state import WorkloadState


def run_client() -> None:
    """Run Flower client."""
    log(INFO, "Long-running Flower client starting")

    args = _parse_args_client().parse_args()

    # Obtain certificates
    if args.insecure:
        if args.root_certificates is not None:
            sys.exit(
                "Conflicting options: The '--insecure' flag disables HTTPS, "
                "but '--root-certificates' was also specified. Please remove "
                "the '--root-certificates' option when running in insecure mode, "
                "or omit '--insecure' to use HTTPS."
            )
        log(WARN, "Option `--insecure` was set. Starting insecure HTTP client.")
        root_certificates = None
    else:
        # Load the certificates if provided, or load the system certificates
        cert_path = args.root_certificates
        if cert_path is None:
            root_certificates = None
        else:
            root_certificates = Path(cert_path).read_bytes()

    print(args.root_certificates)
    print(args.server)
    print(args.callable_dir)
    print(args.callable)

    callable_dir = args.callable_dir
    if callable_dir is not None:
        sys.path.insert(0, callable_dir)

    def _load() -> Flower:
        flower: Flower = load_callable(args.callable)
        return flower

    return start_client(
        server_address=args.server,
        load_callable_fn=_load,
        transport="grpc-rere",  # Only
        root_certificates=root_certificates,
        insecure=args.insecure,
    )


def _parse_args_client() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a long-running Flower client",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the client without HTTPS. By default, the client runs with "
        "HTTPS enabled. Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--root-certificates",
        metavar="ROOT_CERT",
        type=str,
        help="Specifies the path to the PEM-encoded root certificate file for "
        "establishing secure HTTPS connections.",
    )
    parser.add_argument(
        "--server",
        default="0.0.0.0:9092",
        help="Server address",
    )
    parser.add_argument(
        "--callable",
        help="For example: `client:flower` or `project.package.module:wrapper.flower`",
    )
    parser.add_argument(
        "--callable-dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load callable from there."
        " Default: current working directory.",
    )

    return parser


def _check_actionable_client(
    client: Optional[Client], client_fn: Optional[ClientFn]
) -> None:
    if client_fn is None and client is None:
        raise Exception("Both `client_fn` and `client` are `None`, but one is required")

    if client_fn is not None and client is not None:
        raise Exception(
            "Both `client_fn` and `client` are provided, but only one is allowed"
        )


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def start_client(
    *,
    server_address: str,
    load_callable_fn: Optional[Callable[[], Flower]] = None,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    load_callable_fn : Optional[Callable[[], Flower]] (default: None)
        ...
    client_fn : Optional[ClientFn]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : bool (default: True)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)

    Examples
    --------
    Starting a gRPC client with an insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>> )

    Starting an SSL-enabled gRPC client using system certificates:

    >>> def client_fn(cid: str):
    >>>     return FlowerClient()
    >>>
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>>     insecure=False,
    >>> )

    Starting an SSL-enabled gRPC client using provided certificates:

    >>> from pathlib import Path
    >>>
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    event(EventType.START_CLIENT_ENTER)

    if insecure is None:
        insecure = root_certificates is None

    if load_callable_fn is None:
        _check_actionable_client(client, client_fn)

        if client_fn is None:
            # Wrap `Client` instance in `client_fn`
            def single_client_factory(
                cid: str,  # pylint: disable=unused-argument
            ) -> Client:
                if client is None:  # Added this to keep mypy happy
                    raise Exception(
                        "Both `client_fn` and `client` are `None`, but one is required"
                    )
                return client  # Always return the same instance

            client_fn = single_client_factory

        def _load_app() -> Flower:
            return Flower(client_fn=client_fn)

        load_callable_fn = _load_app
    else:
        warn_experimental_feature("`load_callable_fn`")

    # At this point, only `load_callable_fn` should be used
    # Both `client` and `client_fn` must not be used directly

    # Initialize connection context manager
    connection, address = _init_connection(transport, server_address)

    while True:
        sleep_duration: int = 0
        with connection(
            address,
            insecure,
            grpc_max_message_length,
            root_certificates,
        ) as conn:
            receive, send, create_node, delete_node = conn

            # Register node
            if create_node is not None:
                create_node()  # pylint: disable=not-callable

            while True:
                # Receive
                task_ins = receive()
                if task_ins is None:
                    time.sleep(3)  # Wait for 3s before asking again
                    continue

                # Handle control message
                task_res, sleep_duration = handle_control_message(task_ins=task_ins)
                if task_res:
                    send(task_res)
                    break

                # Load app
                app: Flower = load_callable_fn()

                # Handle task message
                fwd_msg: Fwd = Fwd(
                    task_ins=task_ins,
                    state=WorkloadState(state={}),
                )
                bwd_msg: Bwd = app(fwd=fwd_msg)

                # Send
                send(bwd_msg.task_res)

            # Unregister node
            if delete_node is not None:
                delete_node()  # pylint: disable=not-callable

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)

    event(EventType.START_CLIENT_LEAVE)


def start_numpy_client(
    *,
    server_address: str,
    client: NumPyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
) -> None:
    """Start a Flower NumPyClient which connects to a gRPC server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower server runs on
        the same machine on port 8080, then `server_address` would be
        `"[::]:8080"`.
    client : flwr.client.NumPyClient
        An implementation of the abstract base class `flwr.client.NumPyClient`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : bytes (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)

    Examples
    --------
    Starting a gRPC client with an insecure server connection:

    >>> start_numpy_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting an SSL-enabled gRPC client using system certificates:

    >>> start_numpy_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     insecure=False,
    >>> )

    Starting an SSL-enabled gRPC client using provided certificates:

    >>> from pathlib import Path
    >>>
    >>> start_numpy_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    # warnings.warn(
    #     "flwr.client.start_numpy_client() is deprecated and will "
    #     "be removed in a future version of Flower. Instead, pass "
    #     "your client to `flwr.client.start_client()` by calling "
    #     "first the `.to_client()` method as shown below: \n"
    #     "\tflwr.client.start_client(\n"
    #     "\t\tserver_address='<IP>:<PORT>',\n"
    #     "\t\tclient=FlowerClient().to_client()\n"
    #     "\t)",
    #     DeprecationWarning,
    #     stacklevel=2,
    # )

    # Calling this function is deprecated. A warning is thrown.
    # We first need to convert either the supplied client to `Client.`

    wrp_client = client.to_client()

    start_client(
        server_address=server_address,
        client=wrp_client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
    )


def _init_connection(
    transport: Optional[str], server_address: str
) -> Tuple[
    Callable[
        [str, bool, int, Union[bytes, str, None]],
        ContextManager[
            Tuple[
                Callable[[], Optional[TaskIns]],
                Callable[[TaskRes], None],
                Optional[Callable[[], None]],
                Optional[Callable[[], None]],
            ]
        ],
    ],
    str,
]:
    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Set the default transport layer
    if transport is None:
        transport = TRANSPORT_TYPE_GRPC_BIDI

    # Use either gRPC bidirectional streaming or REST request/response
    if transport == TRANSPORT_TYPE_REST:
        try:
            from .rest_client.connection import http_request_response
        except ModuleNotFoundError:
            sys.exit(MISSING_EXTRA_REST)
        if server_address[:4] != "http":
            sys.exit(
                "When using the REST API, please provide `https://` or "
                "`http://` before the server address (e.g. `http://127.0.0.1:8080`)"
            )
        connection = http_request_response
    elif transport == TRANSPORT_TYPE_GRPC_RERE:
        connection = grpc_request_response
    elif transport == TRANSPORT_TYPE_GRPC_BIDI:
        connection = grpc_connection
    else:
        raise ValueError(
            f"Unknown transport type: {transport} (possible: {TRANSPORT_TYPES})"
        )

    return connection, address
