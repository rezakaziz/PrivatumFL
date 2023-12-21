"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import flwr_modif.proto.driver_pb2
import grpc

class DriverStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    CreateWorkload: grpc.UnaryUnaryMultiCallable[
        flwr_modif.proto.driver_pb2.CreateWorkloadRequest,
        flwr_modif.proto.driver_pb2.CreateWorkloadResponse]
    """Request workload_id"""

    GetNodes: grpc.UnaryUnaryMultiCallable[
        flwr_modif.proto.driver_pb2.GetNodesRequest,
        flwr_modif.proto.driver_pb2.GetNodesResponse]
    """Return a set of nodes"""

    PushTaskIns: grpc.UnaryUnaryMultiCallable[
        flwr_modif.proto.driver_pb2.PushTaskInsRequest,
        flwr_modif.proto.driver_pb2.PushTaskInsResponse]
    """Create one or more tasks"""

    PullTaskRes: grpc.UnaryUnaryMultiCallable[
        flwr_modif.proto.driver_pb2.PullTaskResRequest,
        flwr_modif.proto.driver_pb2.PullTaskResResponse]
    """Get task results"""


class DriverServicer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def CreateWorkload(self,
        request: flwr_modif.proto.driver_pb2.CreateWorkloadRequest,
        context: grpc.ServicerContext,
    ) -> flwr_modif.proto.driver_pb2.CreateWorkloadResponse:
        """Request workload_id"""
        pass

    @abc.abstractmethod
    def GetNodes(self,
        request: flwr_modif.proto.driver_pb2.GetNodesRequest,
        context: grpc.ServicerContext,
    ) -> flwr_modif.proto.driver_pb2.GetNodesResponse:
        """Return a set of nodes"""
        pass

    @abc.abstractmethod
    def PushTaskIns(self,
        request: flwr_modif.proto.driver_pb2.PushTaskInsRequest,
        context: grpc.ServicerContext,
    ) -> flwr_modif.proto.driver_pb2.PushTaskInsResponse:
        """Create one or more tasks"""
        pass

    @abc.abstractmethod
    def PullTaskRes(self,
        request: flwr_modif.proto.driver_pb2.PullTaskResRequest,
        context: grpc.ServicerContext,
    ) -> flwr_modif.proto.driver_pb2.PullTaskResResponse:
        """Get task results"""
        pass


def add_DriverServicer_to_server(servicer: DriverServicer, server: grpc.Server) -> None: ...
