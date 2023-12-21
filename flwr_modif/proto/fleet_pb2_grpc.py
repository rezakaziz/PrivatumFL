# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from flwr_modif.proto import fleet_pb2 as flwr_dot_proto_dot_fleet__pb2


class FleetStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateNode = channel.unary_unary(
                '/flwr.proto.Fleet/CreateNode',
                request_serializer=flwr_dot_proto_dot_fleet__pb2.CreateNodeRequest.SerializeToString,
                response_deserializer=flwr_dot_proto_dot_fleet__pb2.CreateNodeResponse.FromString,
                )
        self.DeleteNode = channel.unary_unary(
                '/flwr.proto.Fleet/DeleteNode',
                request_serializer=flwr_dot_proto_dot_fleet__pb2.DeleteNodeRequest.SerializeToString,
                response_deserializer=flwr_dot_proto_dot_fleet__pb2.DeleteNodeResponse.FromString,
                )
        self.PullTaskIns = channel.unary_unary(
                '/flwr.proto.Fleet/PullTaskIns',
                request_serializer=flwr_dot_proto_dot_fleet__pb2.PullTaskInsRequest.SerializeToString,
                response_deserializer=flwr_dot_proto_dot_fleet__pb2.PullTaskInsResponse.FromString,
                )
        self.PushTaskRes = channel.unary_unary(
                '/flwr.proto.Fleet/PushTaskRes',
                request_serializer=flwr_dot_proto_dot_fleet__pb2.PushTaskResRequest.SerializeToString,
                response_deserializer=flwr_dot_proto_dot_fleet__pb2.PushTaskResResponse.FromString,
                )


class FleetServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CreateNode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteNode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PullTaskIns(self, request, context):
        """Retrieve one or more tasks, if possible

        HTTP API path: /api/v1/fleet/pull-task-ins
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PushTaskRes(self, request, context):
        """Complete one or more tasks, if possible

        HTTP API path: /api/v1/fleet/push-task-res
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FleetServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CreateNode': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateNode,
                    request_deserializer=flwr_dot_proto_dot_fleet__pb2.CreateNodeRequest.FromString,
                    response_serializer=flwr_dot_proto_dot_fleet__pb2.CreateNodeResponse.SerializeToString,
            ),
            'DeleteNode': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteNode,
                    request_deserializer=flwr_dot_proto_dot_fleet__pb2.DeleteNodeRequest.FromString,
                    response_serializer=flwr_dot_proto_dot_fleet__pb2.DeleteNodeResponse.SerializeToString,
            ),
            'PullTaskIns': grpc.unary_unary_rpc_method_handler(
                    servicer.PullTaskIns,
                    request_deserializer=flwr_dot_proto_dot_fleet__pb2.PullTaskInsRequest.FromString,
                    response_serializer=flwr_dot_proto_dot_fleet__pb2.PullTaskInsResponse.SerializeToString,
            ),
            'PushTaskRes': grpc.unary_unary_rpc_method_handler(
                    servicer.PushTaskRes,
                    request_deserializer=flwr_dot_proto_dot_fleet__pb2.PushTaskResRequest.FromString,
                    response_serializer=flwr_dot_proto_dot_fleet__pb2.PushTaskResResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'flwr.proto.Fleet', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Fleet(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CreateNode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flwr.proto.Fleet/CreateNode',
            flwr_dot_proto_dot_fleet__pb2.CreateNodeRequest.SerializeToString,
            flwr_dot_proto_dot_fleet__pb2.CreateNodeResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteNode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flwr.proto.Fleet/DeleteNode',
            flwr_dot_proto_dot_fleet__pb2.DeleteNodeRequest.SerializeToString,
            flwr_dot_proto_dot_fleet__pb2.DeleteNodeResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PullTaskIns(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flwr.proto.Fleet/PullTaskIns',
            flwr_dot_proto_dot_fleet__pb2.PullTaskInsRequest.SerializeToString,
            flwr_dot_proto_dot_fleet__pb2.PullTaskInsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PushTaskRes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/flwr.proto.Fleet/PushTaskRes',
            flwr_dot_proto_dot_fleet__pb2.PushTaskResRequest.SerializeToString,
            flwr_dot_proto_dot_fleet__pb2.PushTaskResResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
