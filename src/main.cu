#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <chrono>

// TODO
//      check if file parameter is present
//      create array to reference heater locations instead of recalculating values
//      make use of cuda shared memory
//      restructure project
//      add output to file

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#independent-thread-scheduling

struct block{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    float temp;
};

struct config_values {
    bool is_3d;
    bool padding[3];
    float k;
    uint32_t num_timesteps;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t grid_depth;
    float starting_temp;
    std::vector<block> blocks;
};

block line_to_block(bool is_3d, std::string & line) {
    block out_block;
    std::string token;

    out_block.x = atoi(line.substr(0, line.find(',')).c_str());
    
    token = line.substr(line.find(',') + 1);
    out_block.y = atoi(token.substr(0, token.find(',')).c_str());

    if (is_3d) {
        token = token.substr(token.find(',') + 1);
        out_block.z = atoi(token.substr(0, token.find(',')).c_str());
    } else {
        out_block.z = 0;
    }

    token = token.substr(token.find(',') + 1);
    out_block.width = atoi(token.substr(0, token.find(',')).c_str());
    
    token = token.substr(token.find(',') + 1);
    out_block.height = atoi(token.substr(0, token.find(',')).c_str());

    if (is_3d) {
        token = token.substr(token.find(',') + 1);
        out_block.depth = atoi(token.substr(0, token.find(',')).c_str());
    } else {
        out_block.depth = 1;
    }

    token = token.substr(token.find(',') + 1);
    out_block.temp = atof(token.substr(0, token.find(',')).c_str());

    return out_block;
}

void set_config_values(config_values & conf, std::string & file_name) {
    std::string buf;
    std::string token;
    uint8_t count = 0;

    std::ifstream conf_file(file_name);
    
    if (!conf_file) {
                std::cerr << "Error opening config file.\n";
    } else {
        while (!conf_file.eof()) {
            std::getline(conf_file, buf);
            // Filter out line that dont start with a number
            if(buf[0]== '.' || (buf[0] != '#' && (buf[0] >= '0' && buf[0] <= '9'))) {
            switch (count) {
                case 0:
                    conf.is_3d = (buf[0] == '3');
                    break;
                case 1:
                    conf.k = atof(buf.c_str());
                    break;
                case 2:
                    conf.num_timesteps = atoi(buf.c_str());
                    break;
                case 3: 
                    // GRID SIZE
                    conf.grid_width = atoi(buf.substr(0, buf.find(',')).c_str());
                    
                    token = buf.substr(buf.find(',') + 1);
                    conf.grid_height = atoi(token.substr(0, token.find(',')).c_str());
                
                    if (conf.is_3d) {
                        conf.grid_depth = atoi(token.substr(token.find(',') + 1).c_str());
                    } else {
                        conf.grid_depth = 1;
                    }
                    break;
                case 4:
                    conf.starting_temp = atof(buf.c_str());
                    break;
                default:
                    // PARSE THE REMAIN FILE FOR FIXED BLOCKS
                    conf.blocks.push_back(line_to_block(conf.is_3d, buf));
                    break;    
            }
            ++count;
            }
        }
    }
    conf_file.close();
}

__global__ void init_grid_values(float * a, int size, float value) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size) {
        a[idx] = value;
    }
}

__global__ void copy_array_elements(float * lhs, float * rhs, int size) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size) {
        lhs[idx] = rhs[idx];
    }
}

__global__ void place_fixed_temp_block(float * array, int array_width, int array_height, int x, int y, int z, int w, int h, float value, int size) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));

    // 4d start = x 
    //          + (y * array_width) 
    //          + (z * array_width * array_height)
    //          + (a * array_width * array_height * array_depth)
    // 4d offset = (idx % w) 
    //          + array_width * (idx % (w * h)) / w)
    //          + (array_width * array_height) * ((idx % (w * h * d))  / (w * h))
    //          + (array_width * array_height * array_depth) * (idx / (w * h * d))

    if (idx < size) {
        int start = x + (y * array_width) + (z * array_width * array_height);
        int index = start + (idx % w) + array_width * ((idx % (w * h)) / w) + (array_width * array_height) * (idx / (w * h));
        array[index] = value;
    }
}

__global__ void mono_3d (float * old_grid, float * new_grid, int size, int width, float k, int area) {
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;

    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    float oldValue = old_grid[idx];
    float * newValueLoc = &new_grid[idx];
    //left
    if (idx < size) {
        if (idx % width != 0) {
            // if not out of range and not a left edge;
            *newValueLoc += k * (old_grid[idx - 1] - oldValue);
        }
        //right
        if (idx % width != width - 1) {
            // if not out of range and not a right edge;
            *newValueLoc += k * (old_grid[idx + 1] - oldValue);
        }
        if (idx % area >=  width) {
            // if not out of range and not a top edge;
            *newValueLoc += k * (old_grid[idx - width] - oldValue);
        }
        if (idx % area < area - width) {
            // if not out of range and not a bottom edge;
            *newValueLoc += k * (old_grid[idx + width] - oldValue);
        }
        if (idx >= area) {
            // if not out of range and not a front edge;
            *newValueLoc += k * (old_grid[idx - area] - oldValue);
        }
        if (idx < size - area) {
            // if not out of range and not a back edge;
            *newValueLoc += k * (old_grid[idx + area] - oldValue);
        }
    }

}

__global__ void mono_2d (float * old_grid, float * new_grid, int size, int width, float k, int area) {
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;

    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    float oldValue = old_grid[idx];
    float * newValueLoc = &new_grid[idx];
    //left
    if (idx < size) {
        if (idx % width != 0) {
            // if not out of range and not a left edge;
            *newValueLoc += k * (old_grid[idx - 1] - oldValue);
        }
        //right
        if (idx % width != width - 1) {
            // if not out of range and not a right edge;
            *newValueLoc += k * (old_grid[idx + 1] - oldValue);
        }
        if (idx % area >=  width) {
            // if not out of range and not a top edge;
            *newValueLoc += k * (old_grid[idx - width] - oldValue);
        }
        if (idx % area < area - width) {
            // if not out of range and not a bottom edge;
            *newValueLoc += k * (old_grid[idx + width] - oldValue);
        }
    }

}


int main(int argc, char * argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    config_values conf;
    std::string file_name(argv[1]);
    set_config_values(conf, file_name);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (stop - start);
    std::cout << duration.count() * 1000 * 1000 << "us" << '\n';

    start = std::chrono::high_resolution_clock::now();

    int area = conf.grid_width * conf.grid_height;
    int size = area * conf.grid_depth;
    int TPB = 512;
    int num_blocks = (size + TPB - 1) / TPB;

    float * new_grid;
    float * old_grid;
    float * host_grid = new float[size];

    cudaMalloc((void**) & new_grid, size * sizeof(float));
    cudaMalloc((void**) & old_grid, size * sizeof(float));


    init_grid_values<<<num_blocks, TPB>>>(new_grid, size, conf.starting_temp);

    // Copy fixed values into new_grid
    for (int block_idx = 0; block_idx < conf.blocks.size(); ++block_idx) {
        int block_size = conf.blocks[block_idx].width * conf.blocks[block_idx].height * conf.blocks[block_idx].depth;
        int blocks = (block_size + TPB - 1) / TPB;
        place_fixed_temp_block<<<blocks, TPB>>>(new_grid, conf.grid_width, conf.grid_height,
            conf.blocks[block_idx].x, conf.blocks[block_idx].y, conf.blocks[block_idx].z,
            conf.blocks[block_idx].width, conf.blocks[block_idx].height, conf.blocks[block_idx].temp, block_size);
        cudaThreadSynchronize();
    }

    // calculate fixed values
    

    for (int i = 0; i < conf.num_timesteps; ++i) {
        copy_array_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size);     // old = new
       
        if (conf.is_3d) {
            mono_3d<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
        } else {
            mono_2d<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
        }
        cudaThreadSynchronize();
           // Copy fixed values into new_grid
        for (int block_idx = 0; block_idx < conf.blocks.size(); ++block_idx) {
            int block_size = conf.blocks[block_idx].width * conf.blocks[block_idx].height * conf.blocks[block_idx].depth;
            int blocks = (block_size + TPB - 1) / TPB;
            place_fixed_temp_block<<<blocks, TPB>>>(new_grid, conf.grid_width, conf.grid_height,
                conf.blocks[block_idx].x, conf.blocks[block_idx].y, conf.blocks[block_idx].z,
                conf.blocks[block_idx].width, conf.blocks[block_idx].height, conf.blocks[block_idx].temp, block_size);
            cudaThreadSynchronize();
        }

        cudaThreadSynchronize();
    }

    // Copy elements from GPU to Host
    cudaMemcpy(host_grid, new_grid, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(old_grid);
    cudaFree(new_grid);

    stop = std::chrono::high_resolution_clock::now();
    duration = (stop - start);
    std::cout << duration.count() * 1000 * 1000 << "us" << '\n';

    // Output host_grid values to files and std::cout
    // TODO fix output for last grid element
    int index = 0;
    for (int layer = 0; layer < conf.grid_depth; ++layer) {
        for (int row = 0; row < conf.grid_height; ++row) {
            for(int col = 0; col < conf.grid_width - 1; ++col) {
                std::cout << host_grid[index++] << ',';
            }
            std::cout << host_grid[index++] << '\n';
        }
        std::cout << '\n';
    }

    delete[] host_grid;   

    return 0;
}

 /* Old timestep for loop after copy
        left_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k);
        right_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k);
        top_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
        bottom_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);

        if (conf.is_3d) {
            front_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
            back_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
        }
        */

/*
__global__ void left_elements(float * old_grid, float * new_grid, int size, int width, float k) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /*  USE PTX assembly if nvcc doesn't automatically detect optimal instruction
        https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad
        mad{.hi,.lo,.wide}.type  d, a, b, c;
        mad.hi.sat.s32           d, a, b, c;

        .type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
        
        
        use: @p mad.wide.s32  idx, blockIDx.x, blockDim.x, threadIdx.x;
        uint idx;
        asm("mad.lo.u32  %0, %1, %2, %3"; : "=r"(idx) : "r"(blockIDx.x), "r"(blockDim.x), "r"(threadIdx.x));
    
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));

    * /

    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size && idx % width != 0) {
        // if not out of range and not a left edge;
        new_grid[idx] += k * (old_grid[idx - 1] - old_grid[idx]);
    }
}

__global__ void right_elements(float * old_grid, float * new_grid, int size, int width, float k) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size && idx % width != width - 1) {
        // if not out of range and not a right edge;
        new_grid[idx] += k * (old_grid[idx + 1] - old_grid[idx]);
    }
}

__global__ void top_elements(float * old_grid, float * new_grid, int size, int width, float k, int area) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size && idx % area >=  width) {
        // if not out of range and not a top edge;
        new_grid[idx] += k * (old_grid[idx - width] - old_grid[idx]);
    }
}

__global__ void bottom_elements(float * old_grid, float * new_grid, int size, int width, float k, int area) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size && idx % area < area - width) {
        // if not out of range and not a bottom edge;
        new_grid[idx] += k * (old_grid[idx + width] - old_grid[idx]);
    }
}

__global__ void front_elements(float * old_grid, float * new_grid, int size, int width, float k, int area) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size && idx >= area) {
        // if not out of range and not a front edge;
        new_grid[idx] += k * (old_grid[idx - area] - old_grid[idx]);
    }
}

__global__ void back_elements(float * old_grid, float * new_grid, int size, int width, float k, int area) {
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint idx;
    uint blkID = blockIdx.x;
    uint blkDim = blockDim.x;
    uint thrID = threadIdx.x;
    asm("mad.lo.u32  %0, %1, %2, %3;" : "=r"(idx) : "r"(blkID), "r"(blkDim), "r"(thrID));
    if (idx < size && idx < size - area) {
        // if not out of range and not a back edge;
        new_grid[idx] += k * (old_grid[idx + area] - old_grid[idx]);
    }
}
*/