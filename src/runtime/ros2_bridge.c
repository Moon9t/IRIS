#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <rcl/rcl.h>
#include <rcl/node.h>
#include <rcl/publisher.h>
#include <rcl/subscription.h>
#include <rcl/wait.h>

#include <std_msgs/msg/float64.h>
#include <std_msgs/msg/int64.h>
#include <std_msgs/msg/string.h>
#include <rosidl_runtime_c/string.h>
#include <rosidl_runtime_c/string_functions.h>

#include <geometry_msgs/msg/vector3.h>
#include <geometry_msgs/msg/twist.h>
#include <geometry_msgs/msg/pose.h>

/* ========================================================================== */
/* Legacy Singleton API (backward compat for existing projects)               */
/* ========================================================================== */
static rcl_allocator_t g_allocator;
static rcl_context_t   g_context;
static rcl_node_t*     g_node = NULL;
static rcl_publisher_t* g_pub = NULL;
static int g_initialized = 0;

__declspec(dllexport) int64_t iris_rcl_init(void) {
    fprintf(stderr, "[ros2_bridge] iris_rcl_init called\n");
    if (g_initialized) {
        fprintf(stderr, "[ros2_bridge] already initialized\n");
        return 0;
    }
    g_allocator = rcl_get_default_allocator();
    g_context = rcl_get_zero_initialized_context();
    rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
    rcl_ret_t rc = rcl_init_options_init(&init_options, g_allocator);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] rcl_init_options_init failed: %d\n", rc);
        return -(int64_t)rc;
    }
    rc = rcl_init(0, NULL, &init_options, &g_context);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] rcl_init failed: %d\n", rc);
        return -(int64_t)rc;
    }
    g_initialized = 1;
    fprintf(stderr, "[ros2_bridge] iris_rcl_init completed successfully\n");
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_get_zero_initialized_node(void) {
    fprintf(stderr, "[ros2_bridge] iris_rcl_get_zero_initialized_node called\n");
    if (!g_initialized) return -1;
    if (g_node) {
        fprintf(stderr, "[ros2_bridge] node already exists: %p\n", (void*)g_node);
        return (int64_t)g_node;
    }
    g_node = (rcl_node_t*)malloc(sizeof(rcl_node_t));
    if (!g_node) return -1;
    *g_node = rcl_get_zero_initialized_node();
    rcl_node_options_t opts = rcl_node_get_default_options();
    rcl_ret_t rc = rcl_node_init(g_node, "ai_orchestrator", "robotics", &g_context, &opts);
    if (rc != RCL_RET_OK) {
        free(g_node); g_node = NULL;
        return -1;
    }
    fprintf(stderr, "[ros2_bridge] node initialized successfully: %p\n", (void*)g_node);
    return (int64_t)g_node;
}

__declspec(dllexport) int64_t iris_rcl_get_zero_initialized_publisher(void) {
    fprintf(stderr, "[ros2_bridge] iris_rcl_get_zero_initialized_publisher called\n");
    if (!g_node) return -1;
    if (g_pub) {
        fprintf(stderr, "[ros2_bridge] publisher already exists: %p\n", (void*)g_pub);
        return (int64_t)g_pub;
    }
    g_pub = (rcl_publisher_t*)malloc(sizeof(rcl_publisher_t));
    if (!g_pub) return -1;
    *g_pub = rcl_get_zero_initialized_publisher();
    const rosidl_message_type_support_t* ts = ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64);
    rcl_publisher_options_t pub_opts = rcl_publisher_get_default_options();
    rcl_ret_t rc = rcl_publisher_init(g_pub, g_node, ts, "/orchestrator/actuation", &pub_opts);
    if (rc != RCL_RET_OK) {
        free(g_pub); g_pub = NULL;
        return -1;
    }
    fprintf(stderr, "[ros2_bridge] publisher initialized successfully: %p\n", (void*)g_pub);
    return (int64_t)g_pub;
}

__declspec(dllexport) int64_t iris_rcl_publish(int64_t pub_handle, int64_t scaled_val) {
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    double val = (double)scaled_val / 100000000.0;
    std_msgs__msg__Float64 msg;
    std_msgs__msg__Float64__init(&msg);
    msg.data = val;
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    std_msgs__msg__Float64__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

__declspec(dllexport) int64_t iris_rcl_shutdown(void) {
    fprintf(stderr, "[ros2_bridge] iris_rcl_shutdown called\n");
    if (!g_initialized) return 0;
    if (g_pub) {
        rcl_publisher_fini(g_pub, g_node);
        free(g_pub); g_pub = NULL;
    }
    if (g_node) {
        rcl_node_fini(g_node);
        free(g_node); g_node = NULL;
    }
    rcl_ret_t rc = rcl_shutdown(&g_context);
    g_initialized = 0;
    fprintf(stderr, "[ros2_bridge] shutdown completed with rc = %d\n", rc);
    return rc == RCL_RET_OK ? 0 : -(int64_t)rc;
}

/* ========================================================================== */
/* Type Support Resolver                                                      */
/* ========================================================================== */
static const rosidl_message_type_support_t* resolve_type_support(const char* type_name) {
    if (strcmp(type_name, "std_msgs/msg/Float64") == 0)
        return ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64);
    if (strcmp(type_name, "std_msgs/msg/Int64") == 0)
        return ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int64);
    if (strcmp(type_name, "std_msgs/msg/String") == 0)
        return ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String);
    if (strcmp(type_name, "geometry_msgs/msg/Vector3") == 0)
        return ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Vector3);
    if (strcmp(type_name, "geometry_msgs/msg/Twist") == 0)
        return ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist);
    if (strcmp(type_name, "geometry_msgs/msg/Pose") == 0)
        return ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Pose);
    fprintf(stderr, "[ros2_bridge] unknown message type: %s\n", type_name);
    return NULL;
}

/* ========================================================================== */
/* Dynamic Context API                                                        */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_context_create(void) {
    rcl_context_t* ctx = (rcl_context_t*)malloc(sizeof(rcl_context_t));
    if (!ctx) return 0;
    *ctx = rcl_get_zero_initialized_context();
    rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
    rcl_ret_t rc = rcl_init_options_init(&init_options, rcl_get_default_allocator());
    if (rc != RCL_RET_OK) { free(ctx); return 0; }
    rc = rcl_init(0, NULL, &init_options, ctx);
    if (rc != RCL_RET_OK) { free(ctx); return 0; }
    fprintf(stderr, "[ros2_bridge] context created: %p\n", (void*)ctx);
    return (int64_t)(intptr_t)ctx;
}

__declspec(dllexport) int64_t iris_rcl_context_destroy(int64_t ctx_handle) {
    rcl_context_t* ctx = (rcl_context_t*)(intptr_t)ctx_handle;
    if (!ctx) return -1;
    rcl_ret_t rc = rcl_shutdown(ctx);
    free(ctx);
    fprintf(stderr, "[ros2_bridge] context destroyed, rc=%d\n", rc);
    return rc == RCL_RET_OK ? 0 : -1;
}

/* ========================================================================== */
/* Dynamic Node API                                                           */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_node_create(int64_t ctx_handle, const char* name, const char* ns) {
    rcl_context_t* ctx = (rcl_context_t*)(intptr_t)ctx_handle;
    if (!ctx) return 0;
    rcl_node_t* node = (rcl_node_t*)malloc(sizeof(rcl_node_t));
    if (!node) return 0;
    *node = rcl_get_zero_initialized_node();
    rcl_node_options_t opts = rcl_node_get_default_options();
    rcl_ret_t rc = rcl_node_init(node, name, ns, ctx, &opts);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] node_create failed: %d\n", rc);
        free(node);
        return 0;
    }
    fprintf(stderr, "[ros2_bridge] node '%s' created: %p\n", name, (void*)node);
    return (int64_t)(intptr_t)node;
}

__declspec(dllexport) int64_t iris_rcl_node_destroy(int64_t node_handle) {
    rcl_node_t* node = (rcl_node_t*)(intptr_t)node_handle;
    if (!node) return -1;
    rcl_node_fini(node);
    free(node);
    return 0;
}

/* ========================================================================== */
/* Dynamic Publisher API                                                      */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_publisher_create(int64_t node_handle, const char* topic, const char* msg_type) {
    rcl_node_t* node = (rcl_node_t*)(intptr_t)node_handle;
    if (!node) return 0;
    const rosidl_message_type_support_t* ts = resolve_type_support(msg_type);
    if (!ts) return 0;
    rcl_publisher_t* pub = (rcl_publisher_t*)malloc(sizeof(rcl_publisher_t));
    if (!pub) return 0;
    *pub = rcl_get_zero_initialized_publisher();
    rcl_publisher_options_t pub_opts = rcl_publisher_get_default_options();
    rcl_ret_t rc = rcl_publisher_init(pub, node, ts, topic, &pub_opts);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] publisher_create failed on '%s': %d\n", topic, rc);
        free(pub);
        return 0;
    }
    fprintf(stderr, "[ros2_bridge] publisher '%s' [%s] created: %p\n", topic, msg_type, (void*)pub);
    return (int64_t)(intptr_t)pub;
}

__declspec(dllexport) int64_t iris_rcl_publisher_destroy(int64_t pub_handle, int64_t node_handle) {
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    rcl_node_t* node = (rcl_node_t*)(intptr_t)node_handle;
    if (!pub) return -1;
    rcl_publisher_fini(pub, node);
    free(pub);
    return 0;
}

/* ========================================================================== */
/* Dynamic Subscription API                                                   */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_subscription_create(int64_t node_handle, const char* topic, const char* msg_type) {
    rcl_node_t* node = (rcl_node_t*)(intptr_t)node_handle;
    if (!node) return 0;
    const rosidl_message_type_support_t* ts = resolve_type_support(msg_type);
    if (!ts) return 0;
    rcl_subscription_t* sub = (rcl_subscription_t*)malloc(sizeof(rcl_subscription_t));
    if (!sub) return 0;
    *sub = rcl_get_zero_initialized_subscription();
    rcl_subscription_options_t sub_opts = rcl_subscription_get_default_options();
    rcl_ret_t rc = rcl_subscription_init(sub, node, ts, topic, &sub_opts);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] subscription_create failed on '%s': %d\n", topic, rc);
        free(sub);
        return 0;
    }
    fprintf(stderr, "[ros2_bridge] subscription '%s' [%s] created: %p\n", topic, msg_type, (void*)sub);
    return (int64_t)(intptr_t)sub;
}

__declspec(dllexport) int64_t iris_rcl_subscription_destroy(int64_t sub_handle, int64_t node_handle) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    rcl_node_t* node = (rcl_node_t*)(intptr_t)node_handle;
    if (!sub) return -1;
    rcl_subscription_fini(sub, node);
    free(sub);
    return 0;
}

/* ========================================================================== */
/* Typed Publish Functions                                                    */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_publish_float64(int64_t pub_handle, double val) {
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    std_msgs__msg__Float64 msg;
    std_msgs__msg__Float64__init(&msg);
    msg.data = val;
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    std_msgs__msg__Float64__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

__declspec(dllexport) int64_t iris_rcl_publish_int64(int64_t pub_handle, int64_t val) {
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    std_msgs__msg__Int64 msg;
    std_msgs__msg__Int64__init(&msg);
    msg.data = val;
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    std_msgs__msg__Int64__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

__declspec(dllexport) int64_t iris_rcl_publish_string(int64_t pub_handle, const char* str) {
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    std_msgs__msg__String msg;
    std_msgs__msg__String__init(&msg);
    rosidl_runtime_c__String__assign(&msg.data, str);
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    std_msgs__msg__String__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

__declspec(dllexport) int64_t iris_rcl_publish_vector3(int64_t pub_handle, double x, double y, double z) {
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    geometry_msgs__msg__Vector3 msg;
    geometry_msgs__msg__Vector3__init(&msg);
    msg.x = x; msg.y = y; msg.z = z;
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    geometry_msgs__msg__Vector3__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

__declspec(dllexport) int64_t iris_rcl_publish_twist(int64_t pub_handle,
    double lx, double ly, double lz,
    double ax, double ay, double az)
{
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    geometry_msgs__msg__Twist msg;
    geometry_msgs__msg__Twist__init(&msg);
    msg.linear.x = lx; msg.linear.y = ly; msg.linear.z = lz;
    msg.angular.x = ax; msg.angular.y = ay; msg.angular.z = az;
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    geometry_msgs__msg__Twist__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

__declspec(dllexport) int64_t iris_rcl_publish_pose(int64_t pub_handle,
    double px, double py, double pz,
    double ox, double oy, double oz, double ow)
{
    rcl_publisher_t* pub = (rcl_publisher_t*)(intptr_t)pub_handle;
    if (!pub) return -1;
    geometry_msgs__msg__Pose msg;
    geometry_msgs__msg__Pose__init(&msg);
    msg.position.x = px; msg.position.y = py; msg.position.z = pz;
    msg.orientation.x = ox; msg.orientation.y = oy;
    msg.orientation.z = oz; msg.orientation.w = ow;
    rcl_ret_t rc = rcl_publish(pub, &msg, NULL);
    geometry_msgs__msg__Pose__fini(&msg);
    return rc == RCL_RET_OK ? 0 : -1;
}

/* ========================================================================== */
/* Wait Set — block until a subscription has data or timeout                  */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_wait_for_message(int64_t sub_handle, int64_t ctx_handle, int64_t timeout_ms) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    rcl_context_t* ctx = (rcl_context_t*)(intptr_t)ctx_handle;
    if (!sub || !ctx) return -1;

    rcl_wait_set_t wait_set = rcl_get_zero_initialized_wait_set();
    rcl_ret_t rc = rcl_wait_set_init(&wait_set, 1, 0, 0, 0, 0, 0, ctx, rcl_get_default_allocator());
    if (rc != RCL_RET_OK) return -1;

    rc = rcl_wait_set_clear(&wait_set);
    if (rc != RCL_RET_OK) { rcl_wait_set_fini(&wait_set); return -1; }

    rc = rcl_wait_set_add_subscription(&wait_set, sub, NULL);
    if (rc != RCL_RET_OK) { rcl_wait_set_fini(&wait_set); return -1; }

    int64_t timeout_ns = timeout_ms * 1000000LL;
    rc = rcl_wait(&wait_set, timeout_ns);

    int64_t has_data = 0;
    if (rc == RCL_RET_OK && wait_set.subscriptions[0] != NULL) {
        has_data = 1;
    }
    rcl_wait_set_fini(&wait_set);
    return has_data;
}

/* ========================================================================== */
/* Typed Take (Receive) Functions                                             */
/* ========================================================================== */
__declspec(dllexport) int64_t iris_rcl_take_float64(int64_t sub_handle, double* val_out) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    if (!sub) return 0;
    std_msgs__msg__Float64 msg;
    std_msgs__msg__Float64__init(&msg);
    rcl_ret_t rc = rcl_take(sub, &msg, NULL, NULL);
    if (rc == RCL_RET_OK) {
        *val_out = msg.data;
        std_msgs__msg__Float64__fini(&msg);
        return 1;
    }
    std_msgs__msg__Float64__fini(&msg);
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_take_int64_val(int64_t sub_handle, int64_t* val_out) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    if (!sub) return 0;
    std_msgs__msg__Int64 msg;
    std_msgs__msg__Int64__init(&msg);
    rcl_ret_t rc = rcl_take(sub, &msg, NULL, NULL);
    if (rc == RCL_RET_OK) {
        *val_out = msg.data;
        std_msgs__msg__Int64__fini(&msg);
        return 1;
    }
    std_msgs__msg__Int64__fini(&msg);
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_take_string(int64_t sub_handle, char* buf, int32_t max_len) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    if (!sub) return 0;
    std_msgs__msg__String msg;
    std_msgs__msg__String__init(&msg);
    rcl_ret_t rc = rcl_take(sub, &msg, NULL, NULL);
    if (rc == RCL_RET_OK) {
        if (msg.data.data) {
            strncpy(buf, msg.data.data, max_len - 1);
            buf[max_len - 1] = '\0';
        } else {
            buf[0] = '\0';
        }
        std_msgs__msg__String__fini(&msg);
        return 1;
    }
    std_msgs__msg__String__fini(&msg);
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_take_vector3(int64_t sub_handle, double* out) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    if (!sub) return 0;
    geometry_msgs__msg__Vector3 msg;
    geometry_msgs__msg__Vector3__init(&msg);
    rcl_ret_t rc = rcl_take(sub, &msg, NULL, NULL);
    if (rc == RCL_RET_OK) {
        out[0] = msg.x; out[1] = msg.y; out[2] = msg.z;
        geometry_msgs__msg__Vector3__fini(&msg);
        return 1;
    }
    geometry_msgs__msg__Vector3__fini(&msg);
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_take_twist(int64_t sub_handle, double* out) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    if (!sub) return 0;
    geometry_msgs__msg__Twist msg;
    geometry_msgs__msg__Twist__init(&msg);
    rcl_ret_t rc = rcl_take(sub, &msg, NULL, NULL);
    if (rc == RCL_RET_OK) {
        out[0] = msg.linear.x;  out[1] = msg.linear.y;  out[2] = msg.linear.z;
        out[3] = msg.angular.x; out[4] = msg.angular.y; out[5] = msg.angular.z;
        geometry_msgs__msg__Twist__fini(&msg);
        return 1;
    }
    geometry_msgs__msg__Twist__fini(&msg);
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_take_pose(int64_t sub_handle, double* out) {
    rcl_subscription_t* sub = (rcl_subscription_t*)(intptr_t)sub_handle;
    if (!sub) return 0;
    geometry_msgs__msg__Pose msg;
    geometry_msgs__msg__Pose__init(&msg);
    rcl_ret_t rc = rcl_take(sub, &msg, NULL, NULL);
    if (rc == RCL_RET_OK) {
        out[0] = msg.position.x;    out[1] = msg.position.y;    out[2] = msg.position.z;
        out[3] = msg.orientation.x; out[4] = msg.orientation.y;
        out[5] = msg.orientation.z; out[6] = msg.orientation.w;
        geometry_msgs__msg__Pose__fini(&msg);
        return 1;
    }
    geometry_msgs__msg__Pose__fini(&msg);
    return 0;
}
