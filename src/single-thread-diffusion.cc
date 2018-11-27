#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct block{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t size;
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
    out_block.size = out_block.width * out_block.height * out_block.depth;
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
            buf.erase(std::remove_if( buf.begin(), buf.end(), ::isspace ), buf.end());
            // Filter out line that dont start with a number
            if(buf[0 ]== '.' || (buf[0] != '#' && (buf[0] >= '0' && buf[0] <= '9'))) {
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

void init_grid_values(float * a, int size, float value) {
    for (int i = 0; i < size; ++i){
        a[i] = value;
    }
}

void copy_array_elements(float * lhs, float * rhs, int size) {
    for (int i = 0; i < size; ++i) {
        lhs[i] = rhs[i];
    }
}

void place_fixed_temp_block(float * array, int array_width, int array_height, int x, int y, int z, int w, int h, float value, int size) {
    for (int i = 0; i < size; ++i) {
        int start = x + (y * array_width) + (z * array_width * array_height);
        int index = start + (i % w) + array_width * ((i % (w * h)) / w) + (array_width * array_height) * (i / (w * h));
        array[index] = value;
    }
}

void mono_3d (float * old_grid, float * new_grid, int size, int width, float k, int area) {


    for (int idx = 0; idx < size; ++idx) {
        float oldValue = old_grid[idx];
        float * newValueLoc = &new_grid[idx];

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

void mono_2d (float * old_grid, float * new_grid, int size, int width, float k, int area) {
    for (int idx = 0; idx < size; ++idx) {
        float oldValue = old_grid[idx];
        float * newValueLoc = &new_grid[idx];
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

void copy_fixed_blocks (config_values & conf, int TPB, float *new_grid) {
    // Copy fixed values into new_grid
    for (int block_idx = 0; block_idx < conf.blocks.size(); ++block_idx) {
        place_fixed_temp_block(new_grid, conf.grid_width, conf.grid_height,
            conf.blocks[block_idx].x, conf.blocks[block_idx].y, conf.blocks[block_idx].z,
            conf.blocks[block_idx].width, conf.blocks[block_idx].height, conf.blocks[block_idx].temp, conf.blocks[block_idx].size);
    }
}

void output_final_values (config_values & conf, float * host_grid) {
    std::ofstream out_file("heatOutput.csv");

    int index = 0;
    for (int layer = 0; layer < conf.grid_depth - 1; ++layer) {
        for (int row = 0; row < conf.grid_height; ++row) {
            for(int col = 0; col < conf.grid_width - 1; ++col) {
                out_file << host_grid[index++] << ", ";
            }
            out_file << host_grid[index++] << '\n';
        }
        out_file << '\n';
    }
    for (int row = 0; row < conf.grid_height - 1; ++row) {
        for(int col = 0; col < conf.grid_width - 1; ++col) {
            out_file << host_grid[index++] << ", ";
        }
        out_file << host_grid[index++] << '\n';
    }
    for(int col = 0; col < conf.grid_width - 1; ++col) {
        out_file << host_grid[index++] << ", ";
    }
    out_file << host_grid[index++];
}

int main(int argc, char * argv[]) {

    auto start = std::chrono::high_resolution_clock::now();

    config_values conf;
    std::string file_name(argv[1]);
    set_config_values(conf, file_name);

    int area = conf.grid_width * conf.grid_height;
    int size = area * conf.grid_depth;
    float * new_grid = new float[size];
    float * old_grid = new float[size];

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (stop - start);
    std::cout << duration.count() * 1000 * 1000 << "us" << '\n';

    start = std::chrono::high_resolution_clock::now();

    init_grid_values(new_grid, size, conf.starting_temp);
    copy_fixed_blocks(conf, 0, new_grid);

    if (conf.is_3d) {
        for (int i = 0; i < conf.num_timesteps; ++i) { 
            copy_array_elements(old_grid, new_grid, size);     // old = new
            mono_3d(old_grid, new_grid, size, conf.grid_width, conf.k, area);    
            copy_fixed_blocks(conf, 0, new_grid);

        }
    } else {
        for (int i = 0; i < conf.num_timesteps; ++i) { 
            copy_array_elements(old_grid, new_grid, size);     // old = new
            mono_2d(old_grid, new_grid, size, conf.grid_width, conf.k, area);   
            copy_fixed_blocks(conf, 0, new_grid);
        }
    }

    stop = std::chrono::high_resolution_clock::now();
    duration = (stop - start);
    std::cout << duration.count() * 1000 * 1000 << "us" << '\n';

    output_final_values(conf, new_grid);

    delete[] old_grid;
    delete [] new_grid;

    return 0;
}