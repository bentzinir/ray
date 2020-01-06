#ifndef RAY_RPC_GCS_RPC_CLIENT_H
#define RAY_RPC_GCS_RPC_CLIENT_H

#include "src/ray/rpc/grpc_client.h"

namespace ray {
namespace rpc {

/// Client used for communicating with gcs server.
class GcsRpcClient {
 public:
  /// Constructor.
  ///
  /// \param[in] address Address of gcs server.
  /// \param[in] port Port of the gcs server.
  /// \param[in] client_call_manager The `ClientCallManager` used for managing requests.
  GcsRpcClient(const std::string &address, const int port,
               ClientCallManager &client_call_manager) {
    job_info_grpc_client_ = std::unique_ptr<GrpcClient<JobInfoGcsService>>(
        new GrpcClient<JobInfoGcsService>(address, port, client_call_manager));
    actor_info_grpc_client_ = std::unique_ptr<GrpcClient<ActorInfoGcsService>>(
        new GrpcClient<ActorInfoGcsService>(address, port, client_call_manager));
    node_info_grpc_client_ = std::unique_ptr<GrpcClient<NodeInfoGcsService>>(
        new GrpcClient<NodeInfoGcsService>(address, port, client_call_manager));
    object_info_grpc_client_ = std::unique_ptr<GrpcClient<ObjectInfoGcsService>>(
        new GrpcClient<ObjectInfoGcsService>(address, port, client_call_manager));
    task_info_grpc_client_ = std::unique_ptr<GrpcClient<TaskInfoGcsService>>(
        new GrpcClient<TaskInfoGcsService>(address, port, client_call_manager));
  };

  /// Add job info to gcs server.
  VOID_RPC_CLIENT_METHOD(JobInfoGcsService, AddJob, request, callback,
                         job_info_grpc_client_, )

  /// Mark job as finished to gcs server.
  VOID_RPC_CLIENT_METHOD(JobInfoGcsService, MarkJobFinished, request, callback,
                         job_info_grpc_client_, )

  /// Get actor data from GCS Service.
  VOID_RPC_CLIENT_METHOD(ActorInfoGcsService, GetActorInfo, request, callback,
                         actor_info_grpc_client_, )

  /// Register an actor to GCS Service.
  VOID_RPC_CLIENT_METHOD(ActorInfoGcsService, RegisterActorInfo, request, callback,
                         actor_info_grpc_client_, )

  ///  Update actor info in GCS Service.
  VOID_RPC_CLIENT_METHOD(ActorInfoGcsService, UpdateActorInfo, request, callback,
                         actor_info_grpc_client_, )

  ///  Add actor checkpoint data to GCS Service.
  VOID_RPC_CLIENT_METHOD(ActorInfoGcsService, AddActorCheckpoint, request, callback,
                         actor_info_grpc_client_, )

  ///  Get actor checkpoint data from GCS Service.
  VOID_RPC_CLIENT_METHOD(ActorInfoGcsService, GetActorCheckpoint, request, callback,
                         actor_info_grpc_client_, )

  ///  Get actor checkpoint id data from GCS Service.
  VOID_RPC_CLIENT_METHOD(ActorInfoGcsService, GetActorCheckpointID, request, callback,
                         actor_info_grpc_client_, )

  /// Register a node to GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, RegisterNode, request, callback,
                         node_info_grpc_client_, )

  /// Unregister a node from GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, UnregisterNode, request, callback,
                         node_info_grpc_client_, )

  /// Get information of all nodes from GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, GetAllNodeInfo, request, callback,
                         node_info_grpc_client_, )

  /// Report heartbeat of a node to GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, ReportHeartbeat, request, callback,
                         node_info_grpc_client_, )

  /// Report batch heartbeat to GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, ReportBatchHeartbeat, request, callback,
                         node_info_grpc_client_, )

  /// Get node's resources from GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, GetResources, request, callback,
                         node_info_grpc_client_, )

  /// Update resources of a node in GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, UpdateResources, request, callback,
                         node_info_grpc_client_, )

  /// Delete resources of a node in GCS Service.
  VOID_RPC_CLIENT_METHOD(NodeInfoGcsService, DeleteResources, request, callback,
                         node_info_grpc_client_, )

  /// Get object's locations from GCS Service.
  VOID_RPC_CLIENT_METHOD(ObjectInfoGcsService, GetObjectLocations, request, callback,
                         object_info_grpc_client_, )

  /// Add location of object to GCS Service.
  VOID_RPC_CLIENT_METHOD(ObjectInfoGcsService, AddObjectLocation, request, callback,
                         object_info_grpc_client_, )

  /// Remove location of object to GCS Service.
  VOID_RPC_CLIENT_METHOD(ObjectInfoGcsService, RemoveObjectLocation, request, callback,
                         object_info_grpc_client_, )

  /// Add a task to GCS Service.
  VOID_RPC_CLIENT_METHOD(TaskInfoGcsService, AddTask, request, callback,
                         task_info_grpc_client_, )

  /// Get task information from GCS Service.
  VOID_RPC_CLIENT_METHOD(TaskInfoGcsService, GetTask, request, callback,
                         task_info_grpc_client_, )

  /// Delete tasks from GCS Service.
  VOID_RPC_CLIENT_METHOD(TaskInfoGcsService, DeleteTasks, request, callback,
                         task_info_grpc_client_, )

 private:
  /// The gRPC-generated stub.
  std::unique_ptr<GrpcClient<JobInfoGcsService>> job_info_grpc_client_;
  std::unique_ptr<GrpcClient<ActorInfoGcsService>> actor_info_grpc_client_;
  std::unique_ptr<GrpcClient<NodeInfoGcsService>> node_info_grpc_client_;
  std::unique_ptr<GrpcClient<ObjectInfoGcsService>> object_info_grpc_client_;
  std::unique_ptr<GrpcClient<TaskInfoGcsService>> task_info_grpc_client_;
};

}  // namespace rpc
}  // namespace ray

#endif  // RAY_RPC_GCS_RPC_CLIENT_H
