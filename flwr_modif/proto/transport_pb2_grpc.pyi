"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import flwr_modif.proto.transport_pb2
import grpc
import typing

class FlowerServiceStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    Join: grpc.StreamStreamMultiCallable[
        flwr_modif.proto.transport_pb2.ClientMessage,
        flwr_modif.proto.transport_pb2.ServerMessage]


class FlowerServiceServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def Join(self,
        request_iterator: typing.Iterator[flwr_modif.proto.transport_pb2.ClientMessage],
        context: grpc.ServicerContext,
    ) -> typing.Iterator[flwr_modif.proto.transport_pb2.ServerMessage]: ...


def add_FlowerServiceServicer_to_server(servicer: FlowerServiceServicer, server: grpc.Server) -> None: ...
