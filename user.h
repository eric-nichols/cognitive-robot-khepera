// Synaptic variables
#define USE            0.03   // Usage of synaptic efficacy in range of 0 to 1
#define TAU_REC        130    // (ms) recovery time constant of the synapse
#define TAU_IN         1.5    // (ms) Inactive time constant of the synapse
#define TAU_FACIL      530    // (ms) facilitation time constant
#define L2_WEIGHT      2.8e-9 // layer 2 weight
#define L3_WEIGHT      2.8e-9 // layer 3 weight
#define L4_WEIGHT      6e-9   // layer 4 weight

// LIF Neuron variables
#define R_MEM_BASE 1e+9                       //  (Ohms 1000 * 10^6) Membrane resistence
#define VTH        0.025                      //  (mV) Voltage firing threshold
#define VREST      0.000                      //  (mV)
#define TAU_CONST  -0.01666666666666666643537 // see below
// TAU_MEM   = 60;    //  (ms) Neuron membrane time constant
// Ts        = 1;    // size of delta T for Euler calculations
// TAU_CONST = (Ts / (-1 * tau_mem)  => (1/(-1 * 60))  => -0.01666666666666666643537

// Learning
#define EXPECTINGFRONT 500.5  // between low and high front

// Other
#define TIME_BETWEEN_OUTPUT 40 // time (milliseconds) between output spikes at 25HZ frequency

void layer1(int ms);
void layer2(int ms);
void layer3(int ms);
void layer4(int ms);
void output(struct Robot *robot);
void changed_environment(struct Robot *robot);
boolean previously_experienced_Layer1(boolean modifyVars);
boolean get_new_environment(struct Robot *robot);
void learningRobot(struct Robot *robot, int ms);
boolean front_high();
boolean learning_complete();
//void printLowMedHigh();
