#include "../SRC/include.h"
#include "user_info.h"
#include "user.h"
#include <stdlib.h>   /* Include rand() */
#include <time.h>   // randomness based on time
#include <unistd.h> // for the sleep function
#include <stdio.h>

// declare structures
typedef struct { // structure of synapses
    double weight;
    double x[2];
    double y[2];
    double z[2];
    double Use1[2];
} synapseDATA_struct;

typedef struct { // storage of every synapse
    synapseDATA_struct L1 [16];    // Layer 1 synapses [neuron][synapse]
    synapseDATA_struct L2 [36][3];    // Layer 2 synapses [neuron][synapse]
    synapseDATA_struct L3 [1000][72]; // Layer 3 synapses [neuron][synapse]
    synapseDATA_struct L4 [3][1000];  // Layer 4 synapses [neuron][synapse]
} synapses_struct;

typedef struct { // structure of a membrane
    double Itot[2];
    double volt[2];
} membraneDATA;

typedef struct { // storage of all LIF Membrane voltages and currents
    membraneDATA L1 [16];
    membraneDATA L2 [36];
    membraneDATA L3 [1000];
    membraneDATA L4 [3];
} membranes;

typedef struct { // storage of inter spike interval data for each neuron
    short int  Sensor [6]; // inter spike interval from sensor  to layer 1
    short int  L1 [16];    // inter spike interval from layer 1 to layer 2
    short int  L2 [72];    // inter spike interval from layer 2 to layer 3
    short int  L3 [1000];  // inter spike interval from layer 3 to layer 4
    short int  L4 [3];     // inter spike interval from layer 4 to output
    int        output [2]; // inter spike interval from output (forward/turn)
} inter_spike_interval_struct;

typedef struct { // storage of frequency data - [neuron] [max inputs per neuron]
    float Sensor[12]; // sensor  to layer 1 frequencies [0:5-freqs 6:11-sensor
    float L1 [16];    // sensor  to layer 1 to layer 2 frequencies
    float L2 [72];    // layer 2 to layer 3 frequencies
    float L3 [1000];  // layer 3 to layer 4 frequencies
    float L4 [3];     // layer 4 to output frequencies
} frequencies_struct;

typedef struct { // storage of allowable band passes for each layer
    float L1 [16][2];
    float L2 [2];
    float L3 [2];
    float L4 [2];
} band_passes;

typedef struct { // storage of every synapse
    short int L1 [16];    // Layer 1 synapses [neuron][synapse]
    short int L2 [36][3];    // Layer 2 synapses [neuron][synapse]
    short int L3 [1000][72]; // Layer 3 synapses [neuron][synapse]
    short int L4 [3][1000];  // Layer 4 synapses [neuron][synapse]
} spike_struct;

typedef struct { // storage of every synapse to Layer 2
    short int synapse [3];    // Layer 1 neurons as incoming synapse to neuron [0-35]
} layer2_inputs_struct;

typedef struct { // storage of every synapse to Layer 2
  layer2_inputs_struct l2[36];    // Layer 2 neurons that use layer2_inputs_struct
} layer2_struct;

typedef struct { // storage of every synapse to Layer 2
  short int left[18][8];  // [L2 neuron in region][L1 neuron] Layer 2 left region connections from layer 1
  short int right[18][8]; // [L2 neuron in region][L1 neuron] Layer 2 right region connections from layer 1
} L1_L2_regions_struct;

// initialize structures
synapses_struct synapse;
membranes neuron;
inter_spike_interval_struct  isi;
frequencies_struct frequency;
band_passes band_pass;
spike_struct spike;
layer2_struct layer2_inputs;
L1_L2_regions_struct L1_L2_regions;

// initialize boolean values
boolean observe_reward      = FALSE; // should the reward be observed (during training)?
boolean move                = FALSE; // will there be movement?
boolean environment_changed = TRUE;  // has the environment changed?
boolean AddL3               = TRUE;  // should a layer 3 neuron be added?
boolean experiencedLeft     = TRUE;  // Has the robot experienced this on its left side?
boolean experiencedRight    = TRUE;  // Has the robot experienced this on its right side?

// declare short integer values
short int Num_Neurons    [ 5] = {16,  0,  0,  0, 3}; // layer1=16, layer2L=0, layer2R=0, layer3=0 layer4=3
short int Num_Inputs     [ 5] = { 1,  8,  0,  0, 0}; // layer1=1 layer2L=8 layer3L=0 layer3R=0 layer4=0 to each neuron in layer
short int L2_firing      [ 4] = { 0, 18, 36, 54};    // L2 neurons now firing
short int wheelMove      [ 2] = { 0,  0};            // move a the wheels [left_wheel right_wheel]
short int side_sensors   [ 2];     // if side is left: 0,1  if side is right: 5,6
short int spiking        [ 6];     // our 6 spiking sensors
short int infraredOLD    [12];     // The last infrared values
short int sideDiff       [12];     // difference of the side sensors from the expectation
short int L1_firing      [16];     // from each sensor, which L1 neuron is firing
short int L1_fire        [16];     // temporary storage for comparison against L1_firing
short int modifyWeights  [100][3]; // [L3neuron; forward; turn] [each L3 neuron]
short int learnedL4      [1000][2];// 0=learning 1=done (neuron,1=forward 2=turn)
short int now                = 1;  // will pass between 0 and 1
short int next               = 0;  // will pass between 0 and 1
short int last_L3_spike      = 0;  // last valid spikes from this L3 neuron (using index starting at 0)
short int side               = 0;  // side of the robot that the wall is on. 0=none 1=left 2=right
short int learning           = 0;  // Learning = 0:Not 1:Forward 2:Left 3:Right 4:ForeLeft 5:ForeRight
short int sequence           = 0;  // Learning sequences
short int numL1_L2combos     = 0;  // total number of layer 1 to layer 2 combinations
short int size_this_sequence = 0;  // total number of currently learning neurons
short int lastL4spike        = 0;  // the last layer 4 spike to fire 0=fore, 1=left, 2=right
short int L2syn2spike;             // from Layer 1 spiking, which layer 2 synapses to send the spike through
short int last_output_spike [2] = {0, 0}; // {(0=no 1=yes), (-1=left 0=no 1=right)}
clock_t start;
double end;
// declare and initialize constant short integer values
const short int sense2L1 [16] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5}; // a mapping of sensors to layer 1 neurons
//const short int lamda    [ 3] = {3e-11, 1e-12, 1e-12}; // layer 4 motor learning rate

// {low,mean,high,diff}       Sensor (1 || 6)        Sensor (2 || 5)      Forward (TODO)
//float expecting [3][4] = {{700, 750, 800, 50.0},  {50, 100, 150, 50.0}, {0, 400, 800, 200.0},   }; // middle range sensors
clock_t start;
//double end; // to make the artificial time real
char L2left[18][24];
char L2right[18][24];

int loop = 0;
int robot_location [2][10000000]; // x and y locations for every loop
int output_timer_Fore = 41; // time (milliseconds) since last forward output
int output_timer_Turn = 41; // time (milliseconds) since last turn output

// NOT USED
// writing to file
// FILE *fp, *fpFreqL1, *fpFreqL2L, *fpFreqL2R, *fpFreqL3, *fpFreqL4;
// FILE *fp, *fpFreqL3, *fpFreqL4;
// short int membrane       [16];
// short int layer;                   // layer of network currently working on
// int pas=0;
FILE *network_structure; //*wheelSpeed, *L2toL3, *L3toL4;


/* Learning = 0:Not 1:Forward 2:Left 3:Right 4:ForeLeft 5:ForeRight
   1, No object in front, go forward
   2, Object in front, Stop and turn left
   3, Object in front, Stop and turn right
   4, Forward and left
   5, Forward and right
*/
const short int LearningMove [6][2] = { {0,  0},
                                        {1,  0},
                                        {0, -1},
                                        {0,  1},
                                        {1, -1},
                                        {1,  1} }; // (Learning, direction(forward, turn(Left:-1,Right:1)))


// [direction] [left wheel, right wheel]
//                        no_move,  forward_move, left_move, right_move, forleft_move, forright_move
short int movement [6][2] = { {0, 0}, { 1,  1},  {-1,  1},  { 1, -1},     { 1,  2},     {2,  1} };

// connection data from one layer to the next
short int L1_L2_combinations[16][324]; // (neuron,combination) 324 max combinations
short int L2_L3_combinations[4][1000]; // (L2input_vector, L3neuron)


void DrawStep()
{
  char text[256];
  sprintf(text,"environments encountered: %i",Num_Neurons[3]);
  Color(RED);
  UndrawText(328,150,"OOO");
  Color(BLUE);
  DrawText(120,150,text);
}

void UserInit(struct Robot *robot)
{
    short int neur, syna, tm;

    // layer 1
    for (neur=0; neur<16; neur++)
    {
      synapse.L1[neur].weight = 1.8e-8;    //1.9e-8: 125
      for (tm=0; tm<2; tm++)
      {
        synapse.L1[neur].x[tm]      = 1.0;
        synapse.L1[neur].y[tm]      = 0.0;
        synapse.L1[neur].z[tm]      = 0.0;
        synapse.L1[neur].Use1[tm]   = 0.0;
        neuron.L4[neur].Itot[tm] = 0;
        neuron.L4[neur].volt[tm] = 0;
      }
    } // end for (neur=0; neur<16; neur++)


    // layer 2
    for (neur=0; neur<36; neur++)
    {
      for (syna=0; syna<3;syna++)
      {
        for (tm=0; tm<2; tm++)
        {
          synapse.L2[neur][syna].weight = 0.0;
          synapse.L2[neur][syna].x[tm]      = 1.0;
          synapse.L2[neur][syna].y[tm]      = 0.0;
          synapse.L2[neur][syna].z[tm]      = 0.0;
          synapse.L2[neur][syna].Use1[tm]   = 0.0;
        }
      }

      //initialize neurons
      neuron.L2[neur].Itot[0] = 0;
      neuron.L2[neur].Itot[1] = 0;
      neuron.L2[neur].volt[0] = 0;
      neuron.L2[neur].volt[1] = 0;
    }

    // layer 3
    for (neur=0; neur<1000; neur++)
    {
      for (syna=0; syna<72;syna++)
      {
        for (tm=0; tm<2; tm++)
        {
          synapse.L3[neur][syna].weight = 0.0;
          synapse.L3[neur][syna].x[tm]      = 1.0;
          synapse.L3[neur][syna].y[tm]      = 0.0;
          synapse.L3[neur][syna].z[tm]      = 0.0;
          synapse.L3[neur][syna].Use1[tm]   = 0.0;
        }
      }

      //initialize neurons
      neuron.L3[neur].Itot[0] = 0;
      neuron.L3[neur].Itot[1] = 0;
      neuron.L3[neur].volt[0] = 0;
      neuron.L3[neur].volt[1] = 0;
    }

    // layer 4
    for (neur=0; neur<3; neur++)
    {
      for (syna=0; syna<1000;syna++)
      {
        for (tm=0; tm<2; tm++)
        {
          synapse.L4[neur][syna].weight     = 0.0; //L4_WEIGHT;
          synapse.L4[neur][syna].x[tm]      = 1.0;
          synapse.L4[neur][syna].y[tm]      = 0.0;
          synapse.L4[neur][syna].z[tm]      = 0.0;
          synapse.L4[neur][syna].Use1[tm]   = 0.0;
        } // end of for (tm=0; tm<2; tm++)
      } // end of syna for loop

      //initialize neurons
      neuron.L4[neur].Itot[0] = 0;
      neuron.L4[neur].Itot[1] = 0;
      neuron.L4[neur].volt[0] = 0;
      neuron.L4[neur].volt[1] = 0;
    } // end neur for loop


    // combinations of inputs from layer 2 to layer 3
    L2_L3_combinations[0][0] = 0;
    L2_L3_combinations[1][0] = 18;
    L2_L3_combinations[2][0] = 36;
    L2_L3_combinations[3][0] = 54;

    // sensor to layer 1
    for (neur=0; neur<6; neur++)
      isi.Sensor[neur] = 0;
    for (neur=0; neur<12; neur++)
      frequency.Sensor[neur] = 0.0;

    // layer 1 to 2
    for (neur=0; neur<16; neur++)
    {
      isi.L1[neur] = 0;
      frequency.L1[neur] = 0.0;
      L1_firing[neur] = 0;
    }

    // layer 2 to 3
    for (neur=0; neur<72; neur++)
    {
      isi.L2[neur] = 0;
      frequency.L2[neur] = 0.0;
    }

    // layer 3 to 4
    for (neur=0; neur<1000; neur++)
    {
      isi.L3[neur] = 0;
      frequency.L3[neur] = 0.0;

// might be a runtime bug in this for loop below...
      for (syna=0; syna<2; syna++)
        learnedL4[neur][syna] = 0;
    }

    // layer 4 to output
    for (neur=0; neur<3; neur++)
    {
        isi.L4[neur] = 0;
        frequency.L4[neur] = 0.0;
    }

    // initialize infrared values
    for (neur=0; neur<12; neur++)
    {
        infraredOLD[neur] = 0;
        sideDiff[neur]    = 0;
    }

    frequency.L2[36] = 50.0;
    frequency.L2[54] = 50.0;

// SIDE
// sensor (1, 2, 5, 6): 250 -> 400 mid
// 200 + (0.1 *    1) => 200.1 low
// 200 + (0.1 *  249) => 224.9 low
// 200 + (0.1 *  250) => 225.0 mid
// 200 + (0.1 *  400) => 240.0 mid
// 200 + (0.1 *  401) => 240.1 high
// 200 + (0.1 * 1023) => 302.3 high
//
//  FRONT
//  sensor (3, 4): 1 -> 249 low
// 200 + (0.1 *    1) => 200.1 low
// 200 + (0.1 *  249) => 224.9 low
// 200 + (0.1 *  250) => 230.1 high
// 200 + (0.1 * 1023) => 302.3 high

////  from each of 16 L1 neurons
//// sensor 1
band_pass.L1[0][0] = 200.00; // neuron 1 (low from sensor 1): min
band_pass.L1[0][1] = 224.95; // neuron 1 (low from sensor 1): max
band_pass.L1[1][0] = 224.95; // neuron 2 (med from sensor 1): min
band_pass.L1[1][1] = 240.05; // neuron 2 (med from sensor 1): max
band_pass.L1[2][0] = 240.05; // neuron 3 (high from sensor 1): min
band_pass.L1[2][1] = 302.35; // neuron 3 (high from sensor 1): max

// sensor 2
band_pass.L1[3][0] = 200.00; // neuron 4 (low from sensor 2): min
band_pass.L1[3][1] = 224.95; // neuron 4 (low from sensor 2): max
band_pass.L1[4][0] = 224.95; // neuron 5 (med from sensor 2): min
band_pass.L1[4][1] = 240.05; // neuron 5 (med from sensor 2): max
band_pass.L1[5][0] = 240.05; // neuron 6 (high from sensor 2): min
band_pass.L1[5][1] = 302.35; // neuron 6 (high from sensor 2): max

// sensor 3
band_pass.L1[6][0] = 200.00; // neuron 7 (low from sensor 3): min
band_pass.L1[6][1] = 224.95; // neuron 7 (low from sensor 3): max
band_pass.L1[7][0] = 224.95; // neuron 8 (med from sensor 3): min
band_pass.L1[7][1] = 302.35; // neuron 8 (med from sensor 3): max

// sensor 4
band_pass.L1[8][0] = 200.00; // neuron 9 (low from sensor 4): min
band_pass.L1[8][1] = 224.95; // neuron 9 (low from sensor 4): max
band_pass.L1[9][0] = 224.95; // neuron 10 (med from sensor 4): min
band_pass.L1[9][1] = 302.35; // neuron 10 (med from sensor 4): max

// sensor 5
band_pass.L1[10][0] = 200.00; // neuron 11 (low from sensor 5): min
band_pass.L1[10][1] = 224.95; // neuron 11 (low from sensor 5): max
band_pass.L1[11][0] = 224.95; // neuron 12 (med from sensor 5): min
band_pass.L1[11][1] = 240.05; // neuron 12 (med from sensor 5): max
band_pass.L1[12][0] = 240.05; // neuron 13 (high from sensor 5): min
band_pass.L1[12][1] = 302.35; // neuron 13 (high from sensor 5): max

// sensor 6
band_pass.L1[13][0] = 200.00; // neuron 14 (low from sensor 6): min
band_pass.L1[13][1] = 224.95; // neuron 14 (low from sensor 6): max
band_pass.L1[14][0] = 224.95; // neuron 15 (med from sensor 6): min
band_pass.L1[14][1] = 240.05; // neuron 15 (med from sensor 6): max
band_pass.L1[15][0] = 240.05; // neuron 16 (high from sensor 6): min
band_pass.L1[15][1] = 302.35; // neuron 16 (high from sensor 6): max


band_pass.L2[0] = 95.0;  // has to be 95-295Hz
band_pass.L2[1] = 295.0; // has to be 95-295Hz
band_pass.L3[0] = 47.0;  // has to be 47-295Hz
band_pass.L3[1] = 295.0; // has to be 47-295Hz
band_pass.L4[0] = 47.0;  // has to be 47-295Hz
band_pass.L4[1] = 295.0; // has to be 47-295Hz


// for writing
//fp=fopen("/home/eric/Desktop/output/output.txt", "w");
//fpFreqL1=fopen("/home/eric/Desktop/output/frequenciesL1.txt", "w");
//fpFreqL2L=fopen("/home/eric/Desktop/output/frequenciesL2Left.txt", "w");
//fpFreqL2R=fopen("/home/eric/Desktop/output/frequenciesL2Right.txt", "w");
//fpFreqL3=fopen("/home/eric/Desktop/output/frequenciesL3.txt", "w");
//fpFreqL4=fopen("/home/eric/Desktop/output/frequenciesL4.txt", "w");
//FILE *wheelSpeed, *L2toL3, *L3toL4;
//wheelSpeed = fopen("/home/eric/Desktop/output/wheelSpeed.txt", "w");
//L2toL3 = fopen("/home/eric/Desktop/output/L2toL3.txt", "w");
//L3toL4 = fopen("/home/eric/Desktop/output/L3toL4.txt", "w");
network_structure = fopen("/home/eric/Desktop/network_structure.txt", "w");

}

void UserClose(struct Robot *robot)
{
  //fclose(fp);
  //fclose(fpFreqL1);
  //fclose(fpFreqL2L);
  //fclose(fpFreqL2R);
  //fclose(fpFreqL3);
  //fclose(fpFreqL4);

  //fclose(wheelSpeed);
  //fclose(L2toL3);
  //fclose(L3toL4);
fclose(network_structure);
}

void NewRobot(struct Robot *robot)
{
//  pas = 0;
}

void LoadRobot(struct Robot *robot,FILE *file)
{
}

void SaveRobot(struct Robot *robot,FILE *file)
{
}

void RunRobotStart(struct Robot *robot)
{
  int num;
  ShowUserInfo(2,1);

  Color(GREY_69);
for(num=0;num<loop;num++)
  DrawPoint(((robot_location[0][num]/2)-567),((robot_location[1][num]/2)-162));
loop = 0;
side = 0;
printf("side = 0 \n");
}

void RunRobotStop(struct Robot *robot)
{
  int num;
  ShowUserInfo(1,1);

Color(NAVY_BLUE);
for(num=0;num<loop;num++)
  DrawPoint(((robot_location[0][num]/2)-567),((robot_location[1][num]/2)-162));

fprintf(network_structure, "Number of neurons in Layer 2 Left=%i, Right=%i. Layer 3: %i \n", Num_Neurons[1], Num_Neurons[2], Num_Neurons[3]);

}

boolean StepRobot(struct Robot *robot)
{

  //  pas++;
  //int loop = 0;
//clock_t start, end;
  //double elaps;
  //printf("[x,y] = [%i,%i] \n", robot_location[0][loop], robot_location[1][loop]);

robot_location[0][loop] = robot->X;
robot_location[1][loop] = robot->Y;

  DrawStep();
  loop = loop +1;


if (loop==1)
  start = clock(); // * CLOCKS_PER_SEC; //start the timer


  now = (now +1) % 2;
  next = (next +1) % 2;
  //if (time_to_think < 200)
  //  time_to_think += 1;

  if (learning > 0)  // Is the robot Learning?
    learningRobot(robot, loop);
  else
    environment_changed = get_new_environment(robot); // Now read the updated sensor values (once)

  if (environment_changed)
  {
    robot->Motor[LEFT].Value = 0;  // stop
    robot->Motor[RIGHT].Value = 0; // robot
    //time_to_think = 1; // start thinking about next movement
    changed_environment(robot);
  }

  layer1(loop);
  layer2(loop);
  layer3(loop);
  layer4(loop);
  output(robot);

if (loop%1000==0)
{
  end = (((double)(clock() - start)) / CLOCKS_PER_SEC);
  fprintf(network_structure, "Completed %i artificial seconds in %g seconds with Left:%i, Right:%i, 3rd:%i neurons.\n", (loop/1000), end, Num_Neurons[1], Num_Neurons[2], Num_Neurons[3]);
  //start = clock() * CLOCKS_PER_SEC; //start the timer
}



  //fprintf(wheelSpeed, "%i,%i\n",robot->Motor[LEFT].Value,robot->Motor[RIGHT].Value);
  fflush(stdout); // flush the output stream before sleeping

   return(TRUE);
}

void FastStepRobot(struct Robot *robot)
{
}

void ResetRobot(struct Robot *robot)
{
  //pas = 0;
}

void UserCommand(struct Robot *robot,char *text)
{
  WriteComment("unknown command"); /* no commands */
}

void DrawUserInfo(struct Robot *robot,u_char info,u_char page)
{

  char text[2][256];
  char L3[4];
  char nmb[2];
  int num, xposi;
  sprintf(text[0], "Layer 2                        Layer 3");
  sprintf(text[1], "Left neurons (sensor)     Right neurons (sensor)    #neurons");

  switch(info)
  {
    case 1:
      Color(BLACK);
      DrawText(189,20,text[0]);
      DrawText(18,40,text[1]);
      Color(NAVY_BLUE);
      DrawText(130,40,"sensor");
      DrawText(346,40,"sensor");

      Color(RED);
      DrawLine(429,1,429,337); // separates layer 2 and layer 3
      DrawLine(0,45,500,45);   // horizontal line
      DrawLine(215,30,215,337);// separates layer 2 left and right

      // layer 2 left
      for (num=0, xposi=65; num<Num_Neurons[1]; num++, xposi+=20)
      {
        Color(BLACK);
        sprintf(nmb, "%i", (num +1));
        DrawText(4,xposi,nmb);
        Color(NAVY_BLUE);
        DrawText(28,xposi,L2left[num]);
        Color(RED);
        DrawLine(1,(xposi+3),215,(xposi+3));   // horizontal line
      }

      // layer 2 right
      for (num=0, xposi=65; num<Num_Neurons[2]; num++, xposi+=20)
      {
        sprintf(nmb, "%i", (num +1));
        Color(BLACK);
        DrawText(219,xposi,nmb);
        Color(NAVY_BLUE);
        DrawText(243,xposi,L2right[num]);
        Color(RED);
        DrawLine(215,(xposi+3),429,(xposi+3));   // horizontal line
      }

      Color(BLACK);
      sprintf(L3, "%i", Num_Neurons[3]);
      DrawText(460,65,L3);

//Color(NAVY_BLUE);
//for(num=0;num<loop;num++)
//  DrawPoint(((robot_location[0][num]/2)-567),((robot_location[1][num]/2)-162));
//loop = 0;

      break;
    case 2: DrawStep();
  }
}

void layer1(int ms)
{
  short int cell, n, o;

  // Sensor Input to Layer 1 data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Generate input spike trains according to the input frequencies
  for (n=0; n<6; n++)
  {
    if (isi.Sensor[n] < 1000) // add 1 to isi
      isi.Sensor[n] = isi.Sensor[n] +1;

    // time since last spike >= time between spikes
    if (isi.Sensor[n] >= (1000.0/frequency.Sensor[n]))
    {
      spiking[n] = 1;
      isi.Sensor[n] = 0;
    }
    else
    {
      spiking[n] = 0;
    }
  }


  // for every cell (neuron)
  for (cell=0; cell<16; cell++)
  {

    if (isi.L1[cell] < 1000) // add 1 to isi
      isi.L1[cell] = isi.L1[cell] +1;

    //        stored freqs > current freqs
    if (frequency.L1[cell] > (1000/isi.L1[cell]))// old freqs higher
      frequency.L1[cell] = (1000/isi.L1[cell]); // reduce frequency

    spike.L1[cell] = spiking[sense2L1[cell]]; // generate L1 spike trains: 0s and 1s

    // Excitatory synapse (TS=1, so is removed, as there's no point wasting processor cycles to multiply by 1)
    synapse.L1[cell].x[next] = synapse.L1[cell].x[now]  + (synapse.L1[cell].z[now] / TAU_REC) - ((synapse.L1[cell].Use1[now] * synapse.L1[cell].x[now]) * spike.L1[cell]);
    synapse.L1[cell].y[next] = synapse.L1[cell].y[now]  + (-1 * synapse.L1[cell].y[now] / TAU_IN) + ((synapse.L1[cell].Use1[now] * synapse.L1[cell].x[now]) * spike.L1[cell]);
    synapse.L1[cell].z[next] = synapse.L1[cell].z[now]  + (synapse.L1[cell].y[now] / TAU_IN) - (synapse.L1[cell].z[now] / TAU_REC);

    // facillitating & depressing frequency selection
    if ((frequency.Sensor[sense2L1[cell]] >= band_pass.L1[cell][0]) && (frequency.Sensor[sense2L1[cell]] <= band_pass.L1[cell][1]))
{
      synapse.L1[cell].Use1[next] = synapse.L1[cell].Use1[now]  + (-1 *(synapse.L1[cell].Use1[now] / TAU_FACIL)) + (USE * (1 - synapse.L1[cell].Use1[now]) * spike.L1[cell]);
// if (cell<1)
// fprintf(fp, "At %ims, neuron %i is facilitating. Use1=%g spike=%i ", ms,cell, synapse.L1[cell].Use1[now], spike.L1[cell]);
}
    else
    {
      synapse.L1[cell].Use1[next] = USE;
// if (cell<1)
// fprintf(fp, "At %ims, neuron %i is depressing. Use1=%g spike=%i ", ms,cell,synapse.L1[cell].Use1[now], spike.L1[cell]);
    }

    // Membrane Potential Calculations - amount of current generated on the membrane by the active states of each input
    neuron.L1[cell].Itot[next] = synapse.L1[cell].weight * synapse.L1[cell].y[now]; // only 1 synapse for each neuron
    neuron.L1[cell].volt[next] = neuron.L1[cell].volt[now] + (TAU_CONST * (neuron.L1[cell].volt[now] - (R_MEM_BASE * neuron.L1[cell].Itot[now])));
// if(cell < 1)
// fprintf(fp, "voltage:%g, Itot:%g, weight:%g, y:%g incoming_frequency:%g \n",neuron.L1[cell].volt[now],neuron.L1[cell].Itot[now],synapse.L1[cell].weight,synapse.L1[cell].y[now],frequency.Sensor[sense2L1[cell]]);

    // VARIABLE THRESHOLD
    // Check if the current time step voltage is above the threshold.
    if (neuron.L1[cell].volt[now] >= VTH)
    {
      neuron.L1[cell].Itot[now]   = 0; // now
      neuron.L1[cell].Itot[next]  = 0; // next ms
      neuron.L1[cell].volt[now]   = 0; // now
      neuron.L1[cell].volt[next]  = 0; // next ms

      frequency.L1[cell] = 1000.0 / isi.L1[cell];
      isi.L1[cell] = 0;

//printf("At %ims, a spike from layer 1, neuron %i at frequency %g \n",loop, cell, frequency.L1[cell]);

if (frequency.L1[cell]>90){
if (cell<8)
{
   switch (cell)
  {
    case 0:
    case 1:
    case 2:   L2syn2spike = 0; // Sensor 1 goes to the first synapse
              break;
    case 3:
    case 4:
    case 5:   L2syn2spike = 1; // Sensor 2 goes to the second synapse
              break;
    case 6:
    case 7:   L2syn2spike = 2; // Sensor 3 goes to the third synapse
  }

  for (n=0; n<Num_Neurons[1]; n++)
    if (L1_L2_regions.left[n][cell] == 1)
    {
      spike.L2[n][L2syn2spike] = 1; // spike this layer 2 neuron
//      fprintf(fpFreqL1, "At %ims, a spike from layer 1 neuron %i, to layer 2 neuron %i, synapse %i at frequency %g \n",ms, cell, n, L2syn2spike,frequency.L1[cell]);
    }
}
else // cell >= 8
{
  switch (cell)
  {
    case 8:
    case 9:   L2syn2spike = 0; // Sensor 4 goes to the third synapse
              break;
    case 10:
    case 11:
    case 12:  L2syn2spike = 1; // Sensor 5 goes to the third synapse
              break;
    case 13:
    case 14:
    case 15:  L2syn2spike = 2; // Sensor 6 goes to the third synapse
  }

  for (n=0, o=18; n<Num_Neurons[2]; n++, o++)
    if (L1_L2_regions.right[n][(cell-8)] == 1) {
      spike.L2[o][L2syn2spike] = 1; // spike this layer 2 neuron
//     fprintf(fpFreqL1, "At %ims, a spike from layer 1 neuron %i, to layer 2 neuron %i, synapse %i at frequency %g \n",ms, cell, n, L2syn2spike,frequency.L1[cell]);
    }
}
} // end if frequency is high enough


    }// end if  volt(moderate) >= Vth
  } // end for (cell=0; cell<16; cell++)
}  // end layer1()

void layer2(int ms)
{

  short int cell, s, n, loop;

  // Layer 2 data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for (cell=0; cell<72; cell++) // for each layer 2 cell
  {
    if (isi.L2[cell] < 1000)   // if less than 1000
      isi.L2[cell] += 1;         // update the ISIs by incrementing by 1
  }

  for (cell=0; cell<36; cell++)
    if (frequency.L2[cell] > (1000/isi.L2[cell]))// old freqs higher
      frequency.L2[cell] = (1000/isi.L2[cell]); // reduce frequency

  if (isi.L2[L2_firing[2]] >= (1000./50)) // old value 50hz.
  {
    isi.L2[L2_firing[2]] = 0;
    for (cell=0; cell<Num_Neurons[3]; cell++)         // for each L3 neuron
      if (L2_firing[2]==L2_L3_combinations[2][cell])	// if connected to this L2 neuron
        spike.L3[cell][2] = 1;                        // fire a spike
  }

  if (isi.L2[L2_firing[3]] >= (1000./50)) // old value 50hz.
  {
    isi.L2[L2_firing[3]] = 0;
    for (cell=0; cell<Num_Neurons[3]; cell++)         // for each L3 neuron
      if (L2_firing[3]==L2_L3_combinations[3][cell])	// if connected to this L2 neuron
      spike.L3[cell][3] = 1;                          // fire a spike
  }


  for (cell=0; cell<Num_Neurons[3]; cell++)
  {
    for (loop=2; loop<4; loop++) // for old neurons
    {
      if (L2_L3_combinations[loop][cell] == L2_firing[loop])
      {
        if (isi.L2[L2_firing[loop]] >= (1000./50)) // 50hz.
        {
          spike.L3[cell][L2_firing[loop]] = 1;
          isi.L2[L2_firing[loop]] = 0;
        } // if isi
      } // if combo
    } // for loop
  } // for cell


//fprintf(fp, "Num_Neurons[1, 2] = [%i, %i] \n", Num_Neurons[1], Num_Neurons[2]);

  // for every layer 2 left region cell (neuron)
  for (cell=0; cell<Num_Neurons[1]; cell++)
  {
    neuron.L2[cell].Itot[next] = 0;
    for (s=0; s<3; s++)
    {
      // Excitatory synapse (TS=1, so is removed, as there's no point wasting processor cycles to multiply by 1)
      synapse.L2[cell][s].x[next] = synapse.L2[cell][s].x[now]  + (synapse.L2[cell][s].z[now] / TAU_REC) - ((synapse.L2[cell][s].Use1[now] * synapse.L2[cell][s].x[now]) * spike.L2[cell][s]);
      synapse.L2[cell][s].y[next] = synapse.L2[cell][s].y[now]  + (-1 * synapse.L2[cell][s].y[now] / TAU_IN) + ((synapse.L2[cell][s].Use1[now] * synapse.L2[cell][s].x[now]) * spike.L2[cell][s]);
      synapse.L2[cell][s].z[next] = synapse.L2[cell][s].z[now]  + (synapse.L2[cell][s].y[now] / TAU_IN) - (synapse.L2[cell][s].z[now] / TAU_REC);

//if (spike.L2[cell][s]==1)
//fprintf(fpFreqL2L, "At %ims, a spike to layer 2 neuron %i, synapse %i at frequency %g \n",ms, cell, s, frequency.L1[layer2_inputs.l2[cell].synapse[s]]);


    // facillitating & depressing frequency selection
    if ((frequency.L1[layer2_inputs.l2[cell].synapse[s]] >= band_pass.L2[0]) && (frequency.L1[s] <= band_pass.L2[1]))
    {
      synapse.L2[cell][s].Use1[next] = synapse.L2[cell][s].Use1[now] + (-1 *(synapse.L2[cell][s].Use1[now] / TAU_FACIL)) + (USE * (1 - synapse.L2[cell][s].Use1[now]) * spike.L2[cell][s]);
//      if ((cell<1) && (ms <150))
//      {
//        numbr = (-1 * synapse.L2[cell][s].y[now] / TAU_IN);
//        fprintf(fp, "cell=%i, s=%i, now=%i, y: %g = %g + %g + ", cell, s, now, synapse.L2[cell][s].y[next], synapse.L2[cell][s].y[now], numbr);
//        fprintf(fp, "((%g * %g) * %i)  layer2_input from: %i \n", synapse.L2[cell][s].Use1[now], synapse.L2[cell][s].x[now], spike.L2[cell][s], layer2_inputs.l2[cell].synapse[s]);
//        //fprintf(fp, "y[now(%i), next(%i)] = [%g %g] \n", now, next, synapse.L2[cell][s].y[now], synapse.L2[cell][s].y[next]);
//        fprintf(fp, "At %ims, neuron %i (left), synapse %i, facilitating. spike=%i volt:%g, Itot:%g, weight:%g, y:%g \n", ms,cell, s, spike.L2[cell][s], neuron.L2[cell].volt[now], neuron.L2[cell].Itot[now], synapse.L2[cell][s].weight, synapse.L2[cell][s].y[now]);
//      }
    }
    else
    {
      synapse.L2[cell][s].Use1[next] = USE;
//      if ((cell<1) && (ms <150))
//      {
//        fprintf(fp, "y[now(%i), next(%i)] = [%g %g] \n", now, next, synapse.L2[cell][s].y[now], synapse.L2[cell][s].y[next]);
//        fprintf(fp, "At %ims, neuron %i (left), synapse %i, depressing. spike=%i volt:%g, Itot:%g, weight:%g, y:%g incoming_freq:%g band_pass:[%g %g] \n", ms,cell, s, spike.L2[cell][s], neuron.L2[cell].volt[now], neuron.L2[cell].Itot[now], synapse.L2[cell][s].weight, synapse.L2[cell][s].y[now],frequency.L1[layer2_inputs.l2[cell].synapse[s]] ,band_pass.L2[0],band_pass.L2[1]);
//      }
    }

      // Membrane Potential Calculations - amount of current generated on the membrane by the active states of each input
      neuron.L2[cell].Itot[next] += (synapse.L2[cell][s].weight * synapse.L2[cell][s].y[now]); // only 1 synapse for each neuron
    }

    neuron.L2[cell].volt[next] = neuron.L2[cell].volt[now] + (TAU_CONST * (neuron.L2[cell].volt[now] - (R_MEM_BASE * neuron.L2[cell].Itot[now])));

//if(cell < 2)
//fprintf(fp, "voltage:%g, Itot:%g, weight:%g, y:%g incoming_frequency:%g \n", neuron.L2[cell].volt[now], neuron.L2[cell].Itot[now], synapse.L2[cell][s].weight, synapse.L2[cell][s].y[now],frequency.L1[cell]);


    // VARIABLE THRESHOLD
    // Check if the current time step voltage is above the threshold.
    if (neuron.L2[cell].volt[now] >= VTH)
    {
      neuron.L2[cell].Itot[now]   = 0; // now
      neuron.L2[cell].Itot[next]  = 0; // next ms
      neuron.L2[cell].volt[now]   = 0; // now
      neuron.L2[cell].volt[next]  = 0; // next ms

      //for (n=0; n<Num_Neurons[3]; n++)
      //  spike.L3[n][0] = 1; // spike this layer 2 neuron from left region 0

for (n=0; n<Num_Neurons[3]; n++)		// each L3 neuron
	if (cell==L2_L3_combinations[0][n])	// if connected to this L2 neuron
        	spike.L3[n][0] = 1;			// fire a spike




      frequency.L2[cell] = 1000.0 / isi.L2[cell];
      isi.L2[cell] = 0;


//= print which neuron fired
//if (frequency.L2[cell]>90)
//fprintf(fpFreqL2L, "At %ims, a spike from layer 2 left, neuron %i at frequency %g \n",ms, cell, frequency.L2[cell]);

      // ffprintf('At %ims, a spike from layer 1, neuron %i at frequency %g \n',lpy, neuron, frequency_L1_to_L2(neuron))
    }// end if  volt(moderate) >= Vth
  } // end for (cell=0; cell<16; cell++) left region



  // for every layer 2 right region cell (neuron)
  for (cell=18; cell<(18 +Num_Neurons[2]); cell++)
  {
    neuron.L2[cell].Itot[next] = 0.0;
    for (s=0; s<3; s++)
    {
      // Excitatory synapse (TS=1, so is removed, as there's no point wasting processor cycles to multiply by 1)
      synapse.L2[cell][s].x[next] = synapse.L2[cell][s].x[now]  + (synapse.L2[cell][s].z[now] / TAU_REC) - ((synapse.L2[cell][s].Use1[now] * synapse.L2[cell][s].x[now]) * spike.L2[cell][s]);
      synapse.L2[cell][s].y[next] = synapse.L2[cell][s].y[now]  + (-1 * synapse.L2[cell][s].y[now] / TAU_IN) + ((synapse.L2[cell][s].Use1[now] * synapse.L2[cell][s].x[now]) * spike.L2[cell][s]);
      synapse.L2[cell][s].z[next] = synapse.L2[cell][s].z[now]  + (synapse.L2[cell][s].y[now] / TAU_IN) - (synapse.L2[cell][s].z[now] / TAU_REC);



//if (spike.L2[cell][s]==1)
//fprintf(fpFreqL2R, "At %ims, a spike to layer 2 neuron %i, synapse %i at frequency %g \n",ms, cell, s, frequency.L1[layer2_inputs.l2[cell].synapse[s]]);





      // facillitating & depressing frequency selection
      if ((frequency.L1[layer2_inputs.l2[cell].synapse[s]] >= band_pass.L2[0]) && (frequency.L1[layer2_inputs.l2[cell].synapse[s]] <= band_pass.L2[1]))
      {
        synapse.L2[cell][s].Use1[next] = synapse.L2[cell][s].Use1[now]  + (-1 *(synapse.L2[cell][s].Use1[now] / TAU_FACIL)) + (USE * (1 - synapse.L2[cell][s].Use1[now]) * spike.L2[cell][s]);
// if (cell<2)
//if (ms<100)
//fprintf(fp, "At %ims, neuron %i from synapse %i: facilitating. Use1=%g spike=%i ", ms,cell, s, synapse.L2[cell][s].Use1[now], spike.L2[cell][s]);
      }
      else
      {
        synapse.L2[cell][s].Use1[next] = USE;
        // if (cell<2)
//if (ms<100)
//fprintf(fp, "At %ims, neuron %i from synapse %i: depressing. Use1=%g spike=%i ", ms,cell,s,synapse.L2[cell][s].Use1[now], spike.L2[cell][s]);
      }
//if (ms<100)
//fprintf(fp, "volt:%g, Itot:%g, weight:%g, y:%g incoming_freq:%g \n", neuron.L2[cell].volt[now], neuron.L2[cell].Itot[now], synapse.L2[cell][s].weight, synapse.L2[cell][s].y[now], frequency.L1[layer2_inputs.l2[cell].synapse[s]]);     // frequency.L1[sense2L1[reg]]);

      // Membrane Potential Calculations - amount of current generated on the membrane by the active states of each input
      neuron.L2[cell].Itot[next] += synapse.L2[cell][s].weight * synapse.L2[cell][s].y[now]; // only 1 synapse for each neuron
    }

    neuron.L2[cell].volt[next] = neuron.L2[cell].volt[now] + (TAU_CONST * (neuron.L2[cell].volt[now] - (R_MEM_BASE * neuron.L2[cell].Itot[now])));


    // VARIABLE THRESHOLD
    // Check if the current time step voltage is above the threshold.
    if (neuron.L2[cell].volt[now] >= VTH)
    {
      neuron.L2[cell].Itot[now]   = 0; // now
      neuron.L2[cell].Itot[next]  = 0; // next ms
      neuron.L2[cell].volt[now]   = 0; // now
      neuron.L2[cell].volt[next]  = 0; // next ms

//      for (n=0; n<Num_Neurons[3]; n++)
//        spike.L3[n][1] = 1; // spike this layer 2 neuron from right region 1


for (n=0; n<Num_Neurons[3]; n++)		// each L3 neuron
	if (cell==L2_L3_combinations[1][n])	// if connected to this L2 neuron
        	spike.L3[n][1] = 1;			// fire a spike


      frequency.L2[cell] = 1000.0 / isi.L2[cell];
      isi.L2[cell] = 0;


//= print which neuron fired
//if (frequency.L1[cell]>90)
//fprintf(fpFreqL2R, "At %ims, a spike from layer 2 right, neuron %i at frequency %g \n",ms, cell, frequency.L2[cell]);

      // ffprintf('At %ims, a spike from layer 1, neuron %i at frequency %g \n',lpy, neuron, frequency_L1_to_L2(neuron))
    }// end if  volt(moderate) >= Vth
  } // end for (cell=0; cell<16; cell++) left region

  // now we need to clean up spike.L2
  for (cell=0; cell<36; cell++)
  {
    spike.L2[cell][0] = 0;
    spike.L2[cell][1] = 0;
    spike.L2[cell][2] = 0;
  }

}  // end layer2()

void layer3(int ms)
{
  short int syn, cell, n;

  // for every cell (neuron)
  for (cell=0; cell<Num_Neurons[3]; cell++)
  {
    if (isi.L3[cell] < 1000) // increment isi by 1 if less than 1000
        isi.L3[cell] = isi.L3[cell] +1;  // update the ISIs

    if (frequency.L3[cell] > (1000/isi.L3[cell]))// old freqs higher
      frequency.L3[cell] = (1000/isi.L3[cell]); // reduce frequency

    neuron.L3[cell].Itot[next] = 0;
    for (syn=0; syn<4; syn++)
    {

      // Excitatory synapse (TS=1, so is removed, as there's no point wasting processor cycles to multiply by 1)
      synapse.L3[cell][L2_L3_combinations[syn][cell]].x[next] = synapse.L3[cell][L2_L3_combinations[syn][cell]].x[now]  + (synapse.L3[cell][L2_L3_combinations[syn][cell]].z[now] / TAU_REC) - ((synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[now] * synapse.L3[cell][L2_L3_combinations[syn][cell]].x[now]) * spike.L3[cell][syn]);
      synapse.L3[cell][L2_L3_combinations[syn][cell]].y[next] = synapse.L3[cell][L2_L3_combinations[syn][cell]].y[now]  + (-1 * synapse.L3[cell][L2_L3_combinations[syn][cell]].y[now] / TAU_IN) + ((synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[now] * synapse.L3[cell][L2_L3_combinations[syn][cell]].x[now]) * spike.L3[cell][syn]);
      synapse.L3[cell][L2_L3_combinations[syn][cell]].z[next] = synapse.L3[cell][L2_L3_combinations[syn][cell]].z[now]  + (synapse.L3[cell][L2_L3_combinations[syn][cell]].y[now] / TAU_IN) - (synapse.L3[cell][L2_L3_combinations[syn][cell]].z[now] / TAU_REC);

      // facillitating & depressing frequency selection
      if (frequency.L2[L2_L3_combinations[syn][cell]] >= band_pass.L3[0]) //&& (frequency.L2[L2_L3_combinations[syn][cell]] <= band_pass.L3[1]))
        synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[next] = synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[now]  + (-1 *(synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[now] / TAU_FACIL)) + (USE * (1 - synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[now]) * spike.L3[cell][syn]);
      else
        synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[next] = USE;

      // Membrane Potential Calculations - amount of current generated on the membrane by the active states of each input
      neuron.L3[cell].Itot[next] += synapse.L3[cell][L2_L3_combinations[syn][cell]].weight * synapse.L3[cell][L2_L3_combinations[syn][cell]].y[now]; // only 1 synapse for each neuron

    }

    neuron.L3[cell].volt[next] = neuron.L3[cell].volt[now] + (TAU_CONST * (neuron.L3[cell].volt[now] - (R_MEM_BASE * neuron.L3[cell].Itot[now])));
    //fprintf(fp, "at %ims, layer 3 volt:%g, Vth:%g \n\n",ms,neuron.L3[cell].volt[now], VTH);

    // THRESHOLD
    // Check if the current time step voltage is above the threshold.
    if (neuron.L3[cell].volt[now] >= VTH)
    {
      neuron.L3[cell].Itot[now]   = 0; // now
      neuron.L3[cell].Itot[next]  = 0; // next ms, absolute refractory
      neuron.L3[cell].volt[now]   = 0; // now
      neuron.L3[cell].volt[next]  = 0; // next ms, absolute refractory

      for (n=0; n<3; n++) // each layer 4 neuron
        spike.L4[n][cell] = 1; // spike this layer 2 neuron

      frequency.L3[cell] = 1000.0 / isi.L3[cell];
      isi.L3[cell] = 0;

      //if (time_to_think==200)
        for (n=0; n<size_this_sequence; n++) // each learning sequence
          if (modifyWeights[n][0] == cell)  // this cell is learning
            move = TRUE;

//fprintf(fpFreqL3, "At %ims, a spike from neuron %i at frequency %g with output weights [%g, %g, %g],   last_L3_spike: %i \n",ms, cell, frequency.L3[cell], synapse.L4[0][cell].weight, synapse.L4[1][cell].weight, synapse.L4[2][cell].weight, last_L3_spike);

      // ffprintf('At %ims, a spike from layer 1, neuron %i at frequency %g \n',lpy, neuron, frequency_L1_to_L2(neuron))
    }// end if  volt(moderate) >= Vth
  } // end for  (cell=0; cell<Num_Neurons[3]; cell++)


// clean up spikes
for (cell=0; cell<Num_Neurons[3]; cell++)
  for (syn=0; syn<72; syn++)
    spike.L3[cell][syn] = 0;

} // end void layer3(int ms)

void layer4(int ms)
{
    short int cell, syn;

  // for every cell (neuron)
  for (cell=0; cell<3; cell++)
  {
    if (isi.L4[cell] < 1000) // increment isi by 1 if less than 1000
        isi.L4[cell] = isi.L4[cell] +1;  // update the ISIs

    if (frequency.L4[cell] > (1000/isi.L4[cell])) // old freqs higher
        frequency.L4[cell] = (1000/isi.L4[cell]); // reduce frequency

    neuron.L4[cell].Itot[next] = 0;
    for (syn=0; syn<Num_Neurons[3]; syn++) // for every synapse
    {
      // Excitatory synapse (TS=1, so is removed, as there's no point wasting processor cycles to multiply by 1)
      synapse.L4[cell][syn].x[next] = synapse.L4[cell][syn].x[now]  + (synapse.L4[cell][syn].z[now] / TAU_REC) - ((synapse.L4[cell][syn].Use1[now] * synapse.L4[cell][syn].x[now]) * spike.L4[cell][syn]);
      synapse.L4[cell][syn].y[next] = synapse.L4[cell][syn].y[now]  + (-1 * synapse.L4[cell][syn].y[now] / TAU_IN) + ((synapse.L4[cell][syn].Use1[now] * synapse.L4[cell][syn].x[now]) * spike.L4[cell][syn]);
      synapse.L4[cell][syn].z[next] = synapse.L4[cell][syn].z[now]  + (synapse.L4[cell][syn].y[now] / TAU_IN) - (synapse.L4[cell][syn].z[now] / TAU_REC);

      // facillitating & depressing frequency selection
      if (frequency.L3[syn] >= band_pass.L4[0]) //&& (frequency.L3[syn] <= band_pass.L4[1]))
      {
        synapse.L4[cell][syn].Use1[next] = synapse.L4[cell][syn].Use1[now]  + (-1 *(synapse.L4[cell][syn].Use1[now] / TAU_FACIL)) + (USE * (1 - synapse.L4[cell][syn].Use1[now]) * spike.L4[cell][syn]);

//if ((spike.L4[cell][syn]==1) && (synapse.L4[cell][syn].weight > 0.0))
//{
//fprintf(fpFreqL4,  "at %ims, a spike through synapse %i to neuron %i: with weight: %g, x:%g, y:%g, z:%g to neuron volt: %g \n", ms, syn, cell, synapse.L4[cell][syn].weight, synapse.L4[cell][syn].x[now], synapse.L4[cell][syn].y[now], synapse.L4[cell][syn].z[now], neuron.L4[cell].volt[now]);
//}

      }
      else
        synapse.L4[cell][syn].Use1[next] = 0; //USE;

      // Membrane Potential Calculations - amount of current generated on the membrane by the active states of each input
      neuron.L4[cell].Itot[next] += synapse.L4[cell][syn].weight * synapse.L4[cell][syn].y[now]; // only 1 synapse for each neuron

//if (spike.L4[cell][syn]==1)
//{
//fprintf(fpFreqL4,  "at %ims, a spike through synapse %i to neuron %i: with weight %g \n", ms, syn, cell, synapse.L4[cell][syn].weight);
//fprintf(fp, "weight:%g,\ty:%g,\tfreq_input:%g,\tspike:%i,\tItot:%g \n", synapse.L4[cell][syn].weight, synapse.L4[cell][syn].y[now], frequency.L3[syn], spike.L4[cell][syn], neuron.L4[cell].Itot[next]);
//
////fprintf(fp, "at %ims,\tfrom neuron %i,\tsynapse %i:\t", ms, cell, syn);
////fprintf(fp, "x:%g,\ty:%g,\tz:%g,\tItot:%g\tUse1:%g \n", synapse.L3[cell][L2_L3_combinations[syn][cell]].x[now], synapse.L3[cell][L2_L3_combinations[syn][cell]].y[now], synapse.L3[cell][L2_L3_combinations[syn][cell]].z[now], neuron.L3[cell].Itot[next], synapse.L3[cell][L2_L3_combinations[syn][cell]].Use1[now]);
//}


    } // for every synapse


    neuron.L4[cell].volt[next] = neuron.L4[cell].volt[now] + (TAU_CONST * (neuron.L4[cell].volt[now] - (R_MEM_BASE * neuron.L4[cell].Itot[now])));

//fprintf(fp, "at %ims, volt:%g,\tthreshold:%g \n", ms, neuron.L4[cell].volt[now], VTH);

    // THRESHOLD
    // Check if the current time step voltage is above the threshold.
    if (neuron.L4[cell].volt[now] >= VTH)
    {
      neuron.L4[cell].Itot[now]  = 0; // now
      neuron.L4[cell].Itot[next] = 0; // next ms
      neuron.L4[cell].volt[now]  = 0; // now
      neuron.L4[cell].volt[next] = 0; // next ms

      //      for (n=0; n<3; n++) // each layer 4 neuron
      //        spike.L4[n][cell] = 1; // spike this layer 2 neuron

      frequency.L4[cell] = 1000.0 / isi.L4[cell];
      isi.L4[cell] = 0;

      lastL4spike = cell; // last layer 4 spike to fire
      //fprintf(fpFreqL4, "At %ims, a spike from layer 4, neuron %i at frequency %g \n",ms, cell, frequency.L4[cell]);


      // output
      switch (cell) // spike from which neuron
      {
        case 0: output_timer_Fore = 0; last_output_spike[0] =  1; break; // forward
        case 1: output_timer_Turn = 0; last_output_spike[1] = -1; break; // left
        case 2: output_timer_Turn = 0; last_output_spike[1] =  1; break; // right
      }

//      if (cell>0)
//        printf("spike from neuron: %i, output_timer_Turn: %i \n", cell, output_timer_Turn);

//printf("At %ims, a spike from layer 4, neuron %i at frequency %g \n",loop, cell, frequency.L4[cell]);
    }// end if  volt(moderate) >= Vth
  } // end for (cell=0; cell<3; cell++)


// uptate the output timer
if (output_timer_Fore<TIME_BETWEEN_OUTPUT)
{
  output_timer_Fore = output_timer_Fore + 1;
  //printf("%i mod 4: = %i; \n", output_timer_Fore,(output_timer_Fore%4));
  if (output_timer_Fore%4 == 1)
    if(learning==0)
      move = TRUE;

    //printf("output_timer_Fore: %i \n", output_timer_Fore);
}
else if (last_output_spike[0]==1)
{
  last_output_spike[0]==0; // stop moving forward
}


if (output_timer_Turn<TIME_BETWEEN_OUTPUT)
{
  output_timer_Turn = output_timer_Turn + 1;

  //printf("%i mod 4: = %i; \n", output_timer_Turn,(output_timer_Turn%4));
  if (output_timer_Turn%4 == 1)
    if(learning==0)
      move = TRUE;
}
else if (last_output_spike[1]!=0)
{
  last_output_spike[1]==0; // stop turning
}


// clean up spikes
for (cell=0; cell<3; cell++)
  for (syn=0; syn<Num_Neurons[3]; syn++)
    spike.L4[cell][syn] = 0;


}

void output(struct Robot *robot)
{
  if (move)
  {
    move = FALSE;
    if (learning==0) // not learning
    {
      wheelMove[0] = 1; // go
      wheelMove[1] = 1; // forward
      if (output_timer_Fore%4 == 1)  // time to move forward
      {
        if (output_timer_Turn%4 == 1) // time to turn
        {
          if (last_output_spike[1] == -1)
            wheelMove[1] = 2;                  // go forward and left
          else if (last_output_spike[1] == 1)
            wheelMove[0] = 2;                  // go forward and right
        }
      }
      else
      {
        if (output_timer_Turn%4 == 1) // time to turn
        {
          if (last_output_spike[1] == -1)
            wheelMove[0] = -1;                 // turn left
          else if (last_output_spike[1] == 1)
            wheelMove[1] = -1;                 // turn right
        }
      }
    }
    else
    { // robot's learning (learning!=0)
      wheelMove[0] = movement[learning][0];
      wheelMove[1] = movement[learning][1];
    }

    //printf("wheelMove [%i, %i] \n", wheelMove[0], wheelMove[1]);
    robot->Motor[LEFT].Value = wheelMove[0];  // move
    robot->Motor[RIGHT].Value = wheelMove[1]; // robot
  }
  else
  {
   robot->Motor[LEFT].Value = 0;  // move
   robot->Motor[RIGHT].Value = 0; // robot
  }
  //  fprintf(fp, "At %ims, robot movement [%i %i],  learning = %i  ", loop, robot->Motor[LEFT].Value, robot->Motor[RIGHT].Value, learning);

} // end funct output

void changed_environment(struct Robot *robot)
{
  int neurL1, neurL2, reg;
  boolean experiencedL1;
  environment_changed  = FALSE; // reset
  experiencedL1 = previously_experienced_Layer1(TRUE);

    AddL3 = TRUE;          // reset to add a layer 3 neuron
    if (!experiencedL1) // robot has not experienced this Layer 1 environment before
    {
//      wheelMove[0] = 0; // stop the robot's left wheel
//      wheelMove[1] = 0; // stop the robot's right wheel

// short int Num_Neurons [5] = {16, 0, 0, 0, 3}; // layer1=16, layer2L=0, layer2R=0, layer3=0 layer4=3

      numL1_L2combos = numL1_L2combos +1;                  // add this combo

      // short int L1_L2_combinations[16][324]; // (neuron,combination) 324 max combinations
      for (neurL1=0; neurL1<16; neurL1++)
        L1_L2_combinations[neurL1][numL1_L2combos] = L1_firing[neurL1]; // save this experience

      if (experiencedLeft == FALSE) // not previously experienced this on its left
      {
        for (neurL1=0,neurL2=0; neurL1<8; neurL1++)
        {
          L1_L2_regions.left[Num_Neurons[1]][neurL1] = L1_firing[neurL1];
          if (L1_firing[neurL1]==1) // only 3 should be firing
          {
            synapse.L2[Num_Neurons[1]][neurL2].weight = L2_WEIGHT;      // -1 cuz array starts at 0
            layer2_inputs.l2[Num_Neurons[1]].synapse[neurL2] = neurL1; // -1 cuz array starts at 0
            neurL2++;

            switch (neurL1)
            {
              case 0: sprintf(L2left[Num_Neurons[1]], "(1)Low  "); break;
              case 1: sprintf(L2left[Num_Neurons[1]], "(1)Mid  "); break;
              case 2: sprintf(L2left[Num_Neurons[1]], "(1)High "); break;
              case 3: strcat(L2left[Num_Neurons[1]], "(2)Low  "); break;
              case 4: strcat(L2left[Num_Neurons[1]], "(2)Mid  "); break;
              case 5: strcat(L2left[Num_Neurons[1]], "(2)High "); break;
              case 6: strcat(L2left[Num_Neurons[1]], "(3)Low "); break;
              case 7: strcat(L2left[Num_Neurons[1]], "(3)High"); break;
            }
          }
        }

        printf("%s \n",L2left[Num_Neurons[1]]);
        L2_firing[2] = L2_firing[0] +36; // new to old
        L2_firing[0] = Num_Neurons[1]; // new experience in region // -1 cuz array starts at 0
        Num_Neurons[1] = Num_Neurons[1] + 1; // add new and old left layer 2 neurons
      }

      if (experiencedRight == FALSE) // not previously experienced this on its left
      {
        for (neurL1=8, reg=0, neurL2=0; neurL1<16; neurL1++, reg++) // each layer 1 right neuron
        {
          L1_L2_regions.right[Num_Neurons[2]][reg] = L1_firing[neurL1];
          if (L1_firing[neurL1]==1) // only 3 should be firing
          {
            synapse.L2[(Num_Neurons[2]+18)][neurL2].weight = L2_WEIGHT;
            layer2_inputs.l2[(Num_Neurons[2] +18)].synapse[neurL2] = neurL1;
            neurL2++;

            switch (neurL1)
            {
              case 8: sprintf(L2right[Num_Neurons[2]], "(4)Low  "); break;
              case 9: sprintf(L2right[Num_Neurons[2]], "(4)High "); break;
              case 10: strcat(L2right[Num_Neurons[2]], "(5)Low  "); break;
              case 11: strcat(L2right[Num_Neurons[2]], "(5)Mid  "); break;
              case 12: strcat(L2right[Num_Neurons[2]], "(5)High "); break;
              case 13: strcat(L2right[Num_Neurons[2]], "(6)Low "); break;
              case 14: strcat(L2right[Num_Neurons[2]], "(6)Mid "); break;
              case 15: strcat(L2right[Num_Neurons[2]], "(6)High"); break;
            }
          }
        }
        printf("%s \n",L2right[Num_Neurons[2]]);
        L2_firing[3] = L2_firing[1] +36; // new to old
        L2_firing[1] = Num_Neurons[2] +18; // new experience in region
        Num_Neurons[2] = Num_Neurons[2] + 1; // add new and old left layer 2 neurons
        //for (neurL1=8; neurL1<16; neurL1++) // blink
          //isi.L1[neurL1] = 0;    // reset right ISIs
      }
    }

    else // environment changed & we experienced this sense before but with old sensors???
    {
      for (neurL1=0; neurL1<Num_Neurons[3]; neurL1++) // for every experience
      {
        if ((L2_firing[0] == L2_L3_combinations[0][neurL1]) && \
            (L2_firing[1] == L2_L3_combinations[1][neurL1]) && \
            (L2_firing[2] == L2_L3_combinations[2][neurL1]) && \
            (L2_firing[3] == L2_L3_combinations[3][neurL1])) // we experienced it before!
        {

          last_L3_spike = neurL1; // this L3 neuron will fire within frequency

//printLowMedHigh();



          AddL3 = FALSE; // don't add a layer 3 neuron
        }
      } // end for (neurL1=8; neurL1<16; neurL1++) // for every experience
    } // end if (!experiencedL1)


    for (neurL1=36; neurL1<72; neurL1++)
      if (frequency.L2 [neurL1] == 50.0)
        frequency.L2 [neurL1] = 0; // reset old freqs

    frequency.L2[L2_firing[2]] = 50.0; // updated old freq
    frequency.L2[L2_firing[3]] = 50.0; // updated old freq

    if (AddL3) // add a Layer 3 neuron
    {

      synapse.L3[Num_Neurons[3]][L2_firing[0]].weight = L3_WEIGHT;
      synapse.L3[Num_Neurons[3]][L2_firing[1]].weight = L3_WEIGHT;
      synapse.L3[Num_Neurons[3]][L2_firing[2]].weight = L3_WEIGHT;
      synapse.L3[Num_Neurons[3]][L2_firing[3]].weight = L3_WEIGHT;

      // what neurons are connected:
      L2_L3_combinations[0][Num_Neurons[3]] = L2_firing[0];
      L2_L3_combinations[1][Num_Neurons[3]] = L2_firing[1];
      L2_L3_combinations[2][Num_Neurons[3]] = L2_firing[2];
      L2_L3_combinations[3][Num_Neurons[3]] = L2_firing[3];


//fprintf(L2toL3, "%i, %i, %i, %i\n", L2_firing[0], L2_firing[1], L2_firing[2], L2_firing[3]);

      last_L3_spike = Num_Neurons[3]; // index of this spike (array starts at 0, not 1)

      // adding to Num_Neurons[3] last, because the array index is length-1
      Num_Inputs[2] = Num_Neurons[1]; // L3 input from layer 2 left
      Num_Inputs[3] = Num_Neurons[2];   // L3 input from layer 2 right
      Num_Neurons[3] = Num_Neurons[3] + 1; // add a layer 3 neuron


//ffprintf('At %ims, Num_Neurons{3}: %i \t',lpy,Num_Neurons{3})
//fprintf(fp, "number of neurons = Layer2:[%i %i] Layer3: [%i]   L2_firing: [%i, %i, %i, %i] \n",Num_Neurons[1],Num_Neurons[2], Num_Neurons[3], L2_firing[0], L2_firing[1], L2_firing[2], L2_firing[3] );
//fprintf(fp, "%i:   L2_firing: [%i, %i, %i, %i] \n", Num_Neurons[3], L2_firing[0], L2_firing[1], L2_firing[2], L2_firing[3] );




      // Layer 3 to Layer 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      Num_Inputs[4] = Num_Neurons[3];  // # of L4 inputs is # of L3 neurons

//      ffprintf('Number of layer 2 neurons = Left:%i  Right:%i    layer_3_neurons: %i \n', Num_Neurons{2}(1), Num_Neurons{2}(2),Num_Neurons{3})
//fprintf(fp, "Side = %i. \n", side);
      if (side == 0) // allLow =   1:All_Low     0:At_least_1_Med||High
      {

        if ((L1_firing[0]==0) || (L1_firing[3]==0) || (L1_firing[6]==0) || (L1_firing[8]==0) || (L1_firing[10]==0) || (L1_firing[13]==0)) // sensors 1-6 not low
            {
              if ((frequency.Sensor[0] + frequency.Sensor[1] ) > (frequency.Sensor[4] + frequency.Sensor[5]))
                side = 1; // Wall closer to the left
              else if ((frequency.Sensor[0] + frequency.Sensor[1] ) < (frequency.Sensor[4] + frequency.Sensor[5]))
                side = 2; // Wall closer to the right
              else
              {
                if (frequency.Sensor[2] > frequency.Sensor[3])
                  side = 1; // Wall closer to the left
                else if (frequency.Sensor[2] < frequency.Sensor[3])
                  side = 2; // Wall closer to the right
                else
                {
                  // choose a random side, left or right
                  srand( (unsigned int)time( NULL ) ); // random number is based on current time
                  side = rand() % 2 + 1;              // will be 1 or 2
                }
              }

              if (side == 1)        // closer to the left
              {
                side_sensors[0]=7;
                side_sensors[1]=8;  // Use left sensors
                //fprintf(fp, "Wall is closer on the left. \n\n");
              }
              else                  // closer to the right
              {
                side_sensors[0]=11;
                side_sensors[1]=12; // Use right sensors
                //fprintf(fp, "Wall is closer on the right. \n\n");
              }
printf("side = %i \n",side);
            } // end if any not low
      } // end if (side == 0)

      if (learning > 0) // already in the process of learning
      {                 // while adding a new layer 3 neuron
        modifyWeights[size_this_sequence][0] = Num_Neurons[3] - 1;

        if (front_high())   // front is high
        {
          modifyWeights[size_this_sequence][1] = 0; // not moving forward
          modifyWeights[size_this_sequence][2] = LearningMove[learning][1]; // left or right
        } // end if front == high
        else
        { // front == low. Move forward. Turn in same direction
          modifyWeights[size_this_sequence][1] = 1; // moving forward
          modifyWeights[size_this_sequence][2] = modifyWeights[(size_this_sequence-1)][2]; // same turning sequence
        }
      }
      else // learning==0, need to start a learning sequence
      {   // learning =  0:Not 1:Forward 2:Left 3:Right 4:ForeLeft 5:ForeRight
         // wheelMove = [0 0]; % [left_wheel right_wheel]
        // moving: 12=1mm
        sequence = sequence + 1; // Learning sequences
        // ffprintf('starting learning sequence: %i \n', sequence)
        learning = 1; // default Forward
        //    ForLeft=High         Forward=High         Forward=High         ForRight=High
        if (!front_high()) // front is not high
        {
          if (side==1) // wall is on the left and Front is low
          {
            if (L1_firing[3]==1)      // Front Low, ForLeft Low
              learning = 4;           // Forward and Left
            else if ((L1_firing[5]==1))// || (L1_firing[2]==1)) // Front Low, Side High
              learning = 3;           // Turn Right
            else if (L1_firing[0]==1) // Front Low, ForLeft Med, Left Low
              learning = 5;           // Forward and Right
            else // if (L1_firing[0]==0) // Front Low, ForLeft Med, Left Med or High
              learning = 1;           // Forward
          } // end if side == 1
          else if (side==2) // wall is on the right and front is low
          {
            if (L1_firing[10]==1)      // Front Low, ForRight Low
              learning = 5;            // Forward and Right
            else if ((L1_firing[12]==1))// || (L1_firing[15]==1)) // Front Low, Side High
              learning = 2;            // Turn Left
            else if (L1_firing[13]==1) // Front Low, ForRight Med, Right Low
              learning = 4;            // Forward and Left
            else //if (L1_firing[15]==1)   // Front Low, right side is medium or high
              learning = 1;            // Forward
          }
        } //if (!front_high()) // front is not high

        modifyWeights[0][0] = Num_Neurons[3] - 1;
        if (side!=0)
        {
          modifyWeights[0][1] = LearningMove[learning][0];
          modifyWeights[0][2] = LearningMove[learning][1];
        }
        else
        {
          modifyWeights[0][1] = 1; // go forward
          modifyWeights[0][2] = 0; // go forward
        }
        // ffprintf('right_left_wheel_move = [%i   %i] \n',movement[learning[0]],movement[learning[1]])
      } // end if Learning > 0, // learning sequence already started
      //neuron_to_sequence(Num_Neurons{3}) = sequence; // for each neuron, what sequence it's a part of
      //ffprintf('L3, neuron %i belongs to sequence %i \n', Num_Neurons{3},sequence)
      size_this_sequence++;
    } // end if (AddL3) // add a Layer 3 neuron
} // end function changed_environment

boolean front_high()
{
  boolean high = FALSE;
  if (L1_firing[5]==1) //    ForLeft=High
    high = TRUE;
  if (L1_firing[7]==1) //    Forward=High
    high = TRUE;
  if (L1_firing[9]==1) //    Forward=High
    high = TRUE;
  if (L1_firing[12]==1) //   ForRight=High
    high = TRUE;

  if (high)
  {
    if (side==1)      // wall on left
      learning = 3;   // Stop and turn Right
    else if (side==2) // wall on right
      learning = 2;   // Stop and turn Left
  }
//printf("At %ims, high = %i \n",loop,high);
  return high;
}

boolean previously_experienced_Layer1(boolean modifyVars)
{
  short int neurL2;

    // see if we have experienced this environment on the left
    experiencedLeft = FALSE;
    for (neurL2=0; neurL2<Num_Neurons[1]; neurL2++) // for each neuron on left region of L2
    {
      if ((L1_firing[0] == L1_L2_regions.left[neurL2][0]) && \
          (L1_firing[1] == L1_L2_regions.left[neurL2][1]) && \
          (L1_firing[3] == L1_L2_regions.left[neurL2][3]) && \
          (L1_firing[4] == L1_L2_regions.left[neurL2][4]) && \
          (L1_firing[6] == L1_L2_regions.left[neurL2][6]) )
          {
            experiencedLeft = TRUE;  // previously experienced in region
            if (modifyVars)
            {
              L2_firing[2] = L2_firing[0] +36; // last L2 left side to fire
              L2_firing[0] = neurL2; // previously experienced in region
            }
            break;
          }
    }

    // see if we have experienced this environment on the right
    experiencedRight = FALSE;
    for (neurL2=0; neurL2<Num_Neurons[2]; neurL2++) // for each neuron on right region of L2
    {
      if ((L1_firing[8]  == L1_L2_regions.right[neurL2][0]) && \
          (L1_firing[10] == L1_L2_regions.right[neurL2][2]) && \
          (L1_firing[11] == L1_L2_regions.right[neurL2][3]) && \
          (L1_firing[13] == L1_L2_regions.right[neurL2][5]) && \
          (L1_firing[14] == L1_L2_regions.right[neurL2][6]) )
          {
            experiencedRight = TRUE;  // previously experienced in region
            if (modifyVars)
            {
              L2_firing[3] = L2_firing[1] +36; // last L2 right side to fire
              L2_firing[1] = neurL2 +18; // previously experienced in region
            }
            break;
          }
    }

    // see if we have experienced this situation previously
    if (experiencedLeft && experiencedRight) // experienced this input before
    {
      // see if we experienced left and right together
      for (neurL2=0; neurL2<Num_Neurons[3]; neurL2++)
        if (L2_firing[0] == L2_L3_combinations[0][neurL2])
          if (L2_firing[1] == L2_L3_combinations[1][neurL2])
            return TRUE;
    }
    return FALSE;
}

boolean get_new_environment(struct Robot *robot)
{
    boolean changed_environment = FALSE;
    short int neur;
    for (neur=0; neur<6; neur++)
    {
      infraredOLD[neur]     = frequency.Sensor[neur];     // update old sensor values
      infraredOLD[(neur+6)] = frequency.Sensor[(neur+6)]; // update old sensor values
      frequency.Sensor[(neur+6)] = robot->IRSensor[neur].DistanceValue; // new sensor values
      //frequency.Sensor[neur] = log(frequency.Sensor[(neur+6)]) +295;   // frequency value
      frequency.Sensor[neur] = 200 + (0.1 * frequency.Sensor[(neur+6)]);   // frequency value
    }

    // has the environment changed????
    for (neur=0; neur<16; neur++)
    {
      if ((frequency.Sensor[sense2L1[neur]] > band_pass.L1[neur][0]) && (frequency.Sensor[sense2L1[neur]] < band_pass.L1[neur][1]))
        L1_fire[neur] = 1;
      else
        L1_fire[neur] = 0;

      if (L1_fire[neur] != L1_firing[neur])
      {
        changed_environment = TRUE;
        L1_firing[neur] = L1_fire[neur];
      }
    }
    return changed_environment;
}

void learningRobot(struct Robot *robot, int ms)
{

  short int neurL1, neurL2, matching;

  environment_changed = get_new_environment(robot); // Now read the updated sensor values (multiple times)

  // Time to observe the reward?
  // true if there's a L3 neuron for this input...
  if (environment_changed)
  {

    // see if we have experienced this situation previously
    if (previously_experienced_Layer1(FALSE)) // experienced this input before
    {
      for (neurL1=0; neurL1<Num_Neurons[3]; neurL1++) // for every experience
      {
        matching = 0;
        for (neurL2=0; neurL2<4; neurL2++)
          if ((L2_firing[neurL2] != L2_L3_combinations[neurL2][neurL1]) == 0) // we experienced it before!
            ++matching;

        if (matching==4)
          if ((learnedL4[neurL1][0] + learnedL4[neurL1][1]) == 2)
            observe_reward = TRUE;    // time to observe the reward
      }
    }
  } // end if environment changed

  if (!observe_reward)
    observe_reward = learning_complete();

  // motor reward data
  if (observe_reward)
  {
    observe_reward = FALSE;

//      // Observe forward reward
//      // LOG:1->6 ORIGINAL:7->12 ORIGINAL_FRONT:9->10
//      // deltaFore = sensor_distance(1023-sens) - expected_dstance(1023-144.5)
//      // simplified: deltaFore = (144.5 - sens)
//
//      zeroNeurs = modifyWeightsFore(1,modifyWeightsFore(2,:)==0);  // all possible synapses
//      synapse.weight{4}{1}(1,zeroNeurs) = 0;                      // won't be used, cut the synapse
//
//      // Learning = 0:Not 1:Forward 2:Left 3:Right 4:ForeLeft 5:ForeRight
//      indice = modifyWeightsFore(2,:)==1;
//      if (sum(indice) > 0) // robot has moved forward
//      {
//        deltaFore = (EXPECTINGFRONT - max(infrared(9:10))); // Subtract reward from expectation to get error
//        modNeurs = modifyWeightsFore(1,modifyWeightsFore(2,:)==1); // neurons to modify
//
//        // synapse.weight{4}{1}(1,modNeurs) = synapse.weight{4}{1}(1,modNeurs) + (lamda(1) .* deltaFore); // weight + ('learning rate' * 'error' * 'elgibility trace')
//synapse.weight{4}{1}(1,modNeurs) = 6e-9; // change this to line above!!!!!!!!!!!!!!
//
//        if ((max(synapse.weight{4}{1}(1,modNeurs)) > 5.5e-9) || (min(synapse.weight{4}{1}(1,modNeurs)) < 2.5e-9)) // 2.5e-9 gives an output frequency of 2
//          learnedL4(modifyWeightsFore(1,:),1) = 1;
//
//      } // if sum(indice) > 0 % robot moved forward
//
//
//      // Observe side reward
//      indice = modifyWeightsTurn(2,:)~=0; // Learning = 0:Not 1:Forward 2:Left 3:Right 4:ForeLeft 5:ForeRight
//      if sum(indice) > 0  && side > 0 // robot has turned left or right
//        deltaSide = 0;   // zero is default value if sensors aren't true
//        sideDiff(side_sensors) = [abs(infrared(side_sensors(1))-expectingSide(2)), abs(infrared(side_sensors(2))-expectingSide(2))];
//        if sideDiff(side_sensors(1)) > sideDiff(side_sensors(2))  // vec(1) is further from mid-point
//          if sideDiff(side_sensors(1)) > expectingSide(4) // furthest reading from mid-point is outside mid value
//            deltaSide = sideDiff(side_sensors(1)) .* -hardlims(infrared(side_sensors(1))-infraredOLD(side_sensors(1)));
//          end // end if sideDiff(side_sensors(1)) > expectingSide(4)
//        elseif sideDiff(side_sensors(2)) > expectingSide(4) % furthest reading from mid-point is outside mid value
//          deltaSide = sideDiff(side_sensors(2)) .* -hardlims(infrared(side_sensors(2))-infraredOLD(side_sensors(2)));
//        end   // end if sideDiff(side_sensors(1)) > sideDiff(side_sensors(2))
//        // Update the weight
//        // 2.5e-9 gives an output frequency of 2
//        if sum(modifyWeightsTurn(2,:)==-1)>0 // Turned Left
//          ner = [2 3 -1]; // [current_turn_neuron, other_turn_neuron, direction]
//        else // % Turned Right
//          ner = [3 2 1]; // [current_turn_neuron, other_turn_neuron, direction]
//        end
//        //mod_weights_turn = modifyWeightsTurn
//        cutSynapse = modifyWeightsTurn(2,:)~=ner(3); // synapses to cut
//cutSynapse
//        if (sum(cutSynapse) > 0)
//          zeroNeurs = modifyWeightsTurn(1,cutSynapse)  // all possible synapses
//          synapse.weight{4}{1}(ner(1),zeroNeurs) = 0;  // won't be used, cut the synapse
//        end
//        synapse.weight{4}{1}(ner(2),modifyWeightsTurn(1,:)) = 0;  // won't be used, cut the synapse
//        modNeurs = modifyWeightsTurn(1,~cutSynapse) // neurons to modify
//        //synapse.weight{4}{1}(ner(1),modNeurs) = synapse.weight{4}{1}(ner(1),modNeurs) + (lamda(2) .* deltaSide); % weight + ('learning rate' * 'error' * 'elgibility trace')
//        synapse.weight{4}{1}(ner(1),modNeurs) = 6e-9; // change this to line above!!!!!!!!!!!!!!
//
//        if ((max(synapse.weight{4}{1}(ner(1),modifyWeightsTurn(1,indice))) > 5.5e-9) | ... % 5.4e-9 -> 8.4e-9 => 25.0 output
//            (min(synapse.weight{4}{1}(ner(1),modifyWeightsTurn(1,indice))) < 2.5e-9)) % 2.5e-9 gives an output frequency of 2
//          learnedL4(modifyWeightsTurn(1,:),2) = 1;
//        end
//      else
//
//        synapse.weight{4}{1}(2,modifyWeightsTurn(1,:)) = 0;  // won't be used, cut the synapse
//        synapse.weight{4}{1}(3,modifyWeightsTurn(1,:)) = 0;  // won't be used, cut the synapse
//        learnedL4(modifyWeightsFore(1,:),2) = 1;
//      end // end if sum(indice) > 0  && side > 0 % robot has turned left or right
//
//      // Now tidy up...
//      //synapse.weight{4}{1}(:, (synapse.weight{4}{1}(:,modifyWeightsFore(1,:))<2.5e-9)) = 0; % reset all below THRESH weights to zero
//      //synapse.weight{4}{1}(:, (synapse.weight{4}{1}(:,modifyWeightsFore(1,:))>6.5e-9)) = 6.5e-9; % Max thresh for weights
//
//      Learning = 0;
//      fprintf('At %ims, learning is done. \n\n\n',lpy)
//
//      modifyWeightsFore = [];
//      modifyWeightsTurn = [];
//
//
//    end   // end if observe_reward


//fprintf(fp, "The following layer 3 neurons are done learning: \n");
    // Observe rewards
    for (neurL1=0; neurL1<size_this_sequence; neurL1++) // for each situation with movement
    {
      neurL2 = modifyWeights[neurL1][0]; //fprintf(fp, "neuron %i: ", neurL2); // current neuron


//fprintf(L3toL4, "Layer 3 neuron %i: [", neurL2);

      // Observe forward reward
      if (modifyWeights[neurL1][1] == 1)
      {
        synapse.L4[0][neurL2].weight = L4_WEIGHT;  //fprintf(L3toL4, "0 ");
      }

      // Observe turn reward
      if (modifyWeights[neurL1][2] == -1)
      {
        synapse.L4[1][neurL2].weight = L4_WEIGHT;  //fprintf(L3toL4, "1 ");
      }
      else if (modifyWeights[neurL1][2] == 1)
      {
        synapse.L4[2][neurL2].weight = L4_WEIGHT;  //fprintf(L3toL4, "2 ");
      }
      learnedL4[neurL2][0] = 1;
      learnedL4[neurL2][1] = 1;
//fprintf(L3toL4, "] \n");
//fprintf(fp, "\t modWeights: [%i, %i, %i] \n", modifyWeights[neurL1][0], modifyWeights[neurL1][1], modifyWeights[neurL1][2]);
    } // end for loop

    learning = 0;
    size_this_sequence = 0;
  }   // end if observe_reward

} // end function learningRobot(struct Robot *robot)

boolean learning_complete()
{

  // // {low,mean,high,diff}       Sensor (1 || 6)        Sensor (2 || 5)      Forward (TODO)
  // float expecting [3][4] = {{700, 750, 800, 50.0},  {50, 100, 150, 50.0}, {0, 400, 800, 200.0},   }; // middle range sensors
  // learning =  0:Not 1:Forward 2:Left 3:Right 4:ForeLeft 5:ForeRight
  switch (learning)
  {
    case 1: // Forward
      return TRUE; // immediately observe the reward
      break;

    case 2: // Left
    case 3: // {Left, Right}
      if ((L1_firing[6]==1) && (L1_firing[8]==1)) //  front sensors have changed to low
        return TRUE;        //  time to observe the reward
      break;

    case 4: //  moved Forward and Left
      if (side == 1) //  wall on left (was low)
      {
        if ((L1_firing[0]==0) && (L1_firing[3]==0)) //  Side not low any more
          return TRUE;               //  time to observe the reward
      }
      else if (side == 2) //  wall on right (was high)
      {
        if ((L1_firing[12]==0) && (L1_firing[15]==0)) //  Side not high any more
          return TRUE;               //  time to observe the reward
      }
      break;

    case 5: // moved Forward and Right
      if (side == 1) //  wall on left (was high)
      {
        if ((L1_firing[2]==0) && (L1_firing[5]==0)) //  Side not high any more
          return TRUE;               //  time to observe the reward
      }
      else if (side == 2) //  wall on right (was low)
      {
        if ((L1_firing[10]==0) && (L1_firing[13]==0)) //  Side not low any more
          return TRUE;               //  time to observe the reward
      }
  } // end switch Learning


  return FALSE; // not time to observe the reward
}

//void printLowMedHigh()
//{
// printLowMedHigh();
//}
