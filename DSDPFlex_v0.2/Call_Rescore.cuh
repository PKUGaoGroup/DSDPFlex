/**
 * @file Call_Rescore.cuh
 * @author Dong CW (you@domain.com)
 * @brief Support GNINA rescore
 * @version 0.1
 * @date 2023-11-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <cstdlib>
#include "common.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

class Rescore
{
    struct score_entry
    {
        int pose_number;
        float CNNscore;
        float CNNaffinity;
    };

    // info from main
    std::string receptor;
    std::string ligand_out;
    std::string flex_out;
    std::string out_path;
    std::string temp_path;

    FILE *fp_temp_ligand = NULL;
    FILE *fp_temp_flex = NULL;

public:
    std::string temp_ligand_out;

public:
    std::string temp_flex_out;

    std::vector<score_entry> scores;

public:
    /**
     * @brief initialize from main function
     *
     * @param ligand_out ligand out file
     * @param flex_out flex out file
     */
    void initialize_from_input(std::string receptor, std::string ligand_out, std::string flex_out)
    {
        this->receptor = receptor;
        this->ligand_out = ligand_out;
        this->flex_out = flex_out;

        int dir_ind = ligand_out.find_last_of('/');               // dir/to/Ligand_out.pdbqt
        this->out_path = ligand_out.substr(0, dir_ind);           // dir/to
        this->temp_ligand_out = out_path + "/_temp_ligand.pdbqt"; // dir/to/temp/
        this->temp_flex_out = out_path + "/_temp_flex.pdbqt";
    }
    FILE *fp_temp_ligand_out()
    {
        fp_temp_ligand = fopen(temp_ligand_out.c_str(), "w");
        return fp_temp_ligand;
    }
    FILE *fp_temp_flex_out()
    {
        fp_temp_flex = fopen(temp_flex_out.c_str(), "w");
        return fp_temp_flex;
    }
    void close_files()
    {
        if (fp_temp_ligand != NULL)
            fclose(fp_temp_ligand);
        if (fp_temp_flex != NULL)
            fclose(fp_temp_flex);
    }
    /**
     * @brief split output models and
     *
     * @param input_file_path
     * @param output_file_prefix
     */
    void split_file(std::string input_file_path, std::string output_file_prefix)
    {
        std::ifstream input_file(input_file_path);
        std::string line;
        int file_number = 0;
        bool in_model = false;
        std::ofstream output_file;

        while (std::getline(input_file, line))
        {
            if (line.find("MODEL") != std::string::npos)
            {
                in_model = true;
                if (output_file.is_open())
                {
                    output_file.close();
                }
                file_number++;
                std::string output_file_path = temp_path + output_file_prefix + std::to_string(file_number) + ".txt";
                output_file.open(output_file_path);
            }
            else if (line.find("ENDMDL") != std::string::npos)
            {
                in_model = false;
            }
            else if (in_model)
            {
                output_file << line << std::endl;
            }
        }

        if (output_file.is_open())
        {
            output_file.close();
        }
    }

    void sort_files(std::vector<std::string> file_paths)
    {
        std::vector<std::pair<int, std::string>> file_numbers;

        for (std::string file_path : file_paths)
        {
            std::ifstream input_file(file_path);
            std::string line;
            int number = 0;

            while (getline(input_file, line))
            {
                if (line.find("MODEL") != std::string::npos)
                {
                    number = stoi(line.substr(6));
                    break;
                }
            }

            file_numbers.push_back(make_pair(number, file_path));
        }

        sort(file_numbers.begin(), file_numbers.end());

        for (int i = 0; i < file_numbers.size(); i++)
        {
            std::string file_path = file_numbers[i].second;
            std::string temp_path = "temp_" + std::to_string(i) + ".txt";
            std::ifstream input_file(file_path);
            std::ofstream temp_file(temp_path);
            std::string line;

            while (std::getline(input_file, line))
            {
                temp_file << line << std::endl;
            }

            input_file.close();
            temp_file.close();
        }

        std::ofstream output_file("output.txt");

        for (int i = 0; i < file_numbers.size(); i++)
        {
            std::string temp_path = "temp_" + std::to_string(i) + ".txt";
            std::ifstream temp_file(temp_path);
            std::string line;

            while (getline(temp_file, line))
            {
                output_file << line << std::endl;
            }

            temp_file.close();
        }

        output_file.close();

        for (int i = 0; i < file_numbers.size(); i++)
        {
            std::string temp_path = "temp_" + std::to_string(i) + ".txt";
            remove(temp_path.c_str());
        }
    }

    int run()
    {
        std::string input_file_path = "input.txt";
        std::string output_file_prefix = "output_";
        split_file(input_file_path, output_file_prefix);

        std::vector<std::string> file_paths;

        for (int i = 1;; i++)
        {
            std::string file_path = output_file_prefix + std::to_string(i) + ".txt";

            if (!std::ifstream(file_path).good())
            {
                break;
            }

            file_paths.push_back(file_path);
        }

        sort_files(file_paths);

        return 0;
    }

public:
    float call_gnina()
    {
        FILE *fp = NULL;
        char buffer[512];
        std::string cmdline = "gnina --score_only -r " + receptor + " -l " + temp_ligand_out + " --flex " + temp_flex_out;
        // call GNINA rescore
        fp = popen(cmdline.c_str(), "r");
        float CNNscore = 0;
        float CNNaffinity = 0;
        if (fp != NULL)
        {
            while (fgets(buffer, 512, fp) != NULL)
            {
                if (strstr(buffer, "CNNscore") != NULL)
                {
                    char temp[10];
                    sscanf(buffer, "%s %f", temp, &CNNscore);
                    printf("CNNscore %f \n", CNNscore);
                }
            }
        }
        pclose(fp);
        return CNNscore;
    }
};
