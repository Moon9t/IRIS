#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include <rcl/rcl.h>
#include <rcl/node.h>
#include <rcl/publisher.h>
#include <std_msgs/msg/float64.h>

// Global ROS2 structures
static rcl_allocator_t g_allocator;
static rcl_context_t g_context;
static rcl_node_t* g_node = NULL;
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
        fprintf(stderr, "[ros2_bridge] rcl_init_options_init failed with rc = %d\n", rc);
        return -rc;
    }
    
    rc = rcl_init(0, NULL, &init_options, &g_context);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] rcl_init failed with rc = %d\n", rc);
        return -rc;
    }
    
    g_initialized = 1;
    fprintf(stderr, "[ros2_bridge] iris_rcl_init completed successfully\n");
    return 0;
}

__declspec(dllexport) int64_t iris_rcl_get_zero_initialized_node(void) {
    fprintf(stderr, "[ros2_bridge] iris_rcl_get_zero_initialized_node called\n");
    if (!g_initialized) {
        fprintf(stderr, "[ros2_bridge] cannot create node, ROS2 context not initialized\n");
        return -1;
    }
    if (g_node) {
        fprintf(stderr, "[ros2_bridge] node already exists: %p\n", (void*)g_node);
        return (int64_t)g_node;
    }
    
    g_node = (rcl_node_t*)malloc(sizeof(rcl_node_t));
    if (!g_node) {
        fprintf(stderr, "[ros2_bridge] node malloc failed\n");
        return -1;
    }
    *g_node = rcl_get_zero_initialized_node();
    
    rcl_node_options_t node_options = rcl_node_get_default_options();
    rcl_ret_t rc = rcl_node_init(g_node, "ai_orchestrator", "robotics", &g_context, &node_options);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] rcl_node_init failed with rc = %d\n", rc);
        free(g_node);
        g_node = NULL;
        return -1;
    }
    
    fprintf(stderr, "[ros2_bridge] node initialized successfully: %p\n", (void*)g_node);
    return (int64_t)g_node;
}

__declspec(dllexport) int64_t iris_rcl_get_zero_initialized_publisher(void) {
    fprintf(stderr, "[ros2_bridge] iris_rcl_get_zero_initialized_publisher called\n");
    if (!g_node) {
        fprintf(stderr, "[ros2_bridge] cannot create publisher, node not initialized\n");
        return -1;
    }
    if (g_pub) {
        fprintf(stderr, "[ros2_bridge] publisher already exists: %p\n", (void*)g_pub);
        return (int64_t)g_pub;
    }
    
    g_pub = (rcl_publisher_t*)malloc(sizeof(rcl_publisher_t));
    if (!g_pub) {
        fprintf(stderr, "[ros2_bridge] publisher malloc failed\n");
        return -1;
    }
    *g_pub = rcl_get_zero_initialized_publisher();
    
    const rosidl_message_type_support_t* ts = ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float64);
    rcl_publisher_options_t pub_opts = rcl_publisher_get_default_options();
    
    rcl_ret_t rc = rcl_publisher_init(g_pub, g_node, ts, "/orchestrator/actuation", &pub_opts);
    if (rc != RCL_RET_OK) {
        fprintf(stderr, "[ros2_bridge] rcl_publisher_init failed with rc = %d\n", rc);
        free(g_pub);
        g_pub = NULL;
        return -1;
    }
    
    fprintf(stderr, "[ros2_bridge] publisher initialized successfully: %p\n", (void*)g_pub);
    return (int64_t)g_pub;
}

__declspec(dllexport) int64_t iris_rcl_publish(int64_t pub_handle, int64_t scaled_val) {
    rcl_publisher_t* pub = (rcl_publisher_t*)pub_handle;
    if (!pub) {
        return -1;
    }
    
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
    if (!g_initialized) {
        return 0;
    }
    
    if (g_pub) {
        rcl_publisher_fini(g_pub, g_node);
        free(g_pub);
        g_pub = NULL;
    }
    
    if (g_node) {
        rcl_node_fini(g_node);
        free(g_node);
        g_node = NULL;
    }
    
    rcl_ret_t rc = rcl_shutdown(&g_context);
    g_initialized = 0;
    fprintf(stderr, "[ros2_bridge] shutdown completed with rc = %d\n", rc);
    return rc == RCL_RET_OK ? 0 : -rc;
}
