#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <chrono>

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
            if(buf[0] != '#' && (buf[0] >= '0' && buf[0] <= '9')) {
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


int main(int argc, char * argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    config_values conf;
    std::string file_name(argv[1]);
    set_config_values(conf, file_name);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (stop - start);
    std::cout << duration.count() * 1000 * 1000 << "us" << '\n';

    std::cout << "is 3d: " << conf.is_3d << '\n';
    std::cout << "k: " << conf.k << '\n';
    std::cout << "num_timesteps: " << conf.num_timesteps << '\n';
    std::cout << "grid_width: " << conf.grid_width << '\n';
    std::cout << "grid_height: " << conf.grid_height << '\n';
    std::cout << "grid_depth: " << conf.grid_depth << '\n';
    std::cout << "starting_temp: " << conf.starting_temp << '\n';
    std::cout << "block 1 x: " << conf.blocks[0].temp << std::endl;
/*
    //// Have values from config file

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

    // TODO copy fixed values into new_grid

    for (int i = 0; i < conf.num_timesteps; ++i) {
        copy_array_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size);     // old = new

        left_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k);
        right_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k);
        top_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
        bottom_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);

        if (conf.is_3d) {
            front_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
            back_elements<<<num_blocks, TPB>>>(old_grid, new_grid, size, conf.grid_width, conf.k, area);
        }

        //TODO copy fixed values into new_grid

        cudaThreadSynchronize();
    }

    // Copy elements from GPU to Host
    cudaMemcpy(host_grid, new_grid, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(old_grid);
    cudaFree(new_grid);

    // Output host_grid values to files and std::cout
    int index = 0;
    for (int layer = 0; layer < conf.grid_depth; ++layer) {
        for (int row = 0; row < conf.grid_height; ++col) {
            for(int col = 0; col < conf.grid_width - 1) {
                std::cout << host_grid[index++] << ',';
            }
            std::cout << host_grid[index++] << '\n';
        }
        std::count << '\n';
    }


    delete[] host_grid;   
*/ 
    return 0;
}



