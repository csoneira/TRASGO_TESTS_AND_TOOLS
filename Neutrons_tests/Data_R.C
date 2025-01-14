{
//////////////////////////////////////////////////////////
//   This file has been automatically generated 
//     (Thu Feb 15 19:07:43 2024 by ROOT version6.30/04)
//   from TTree Data_R/CoMPASS RAW events TTree
//   found on file: SDataR_Neutrones_4,5-12-21_new+-12V_TestZipiZape_fondo.root
//////////////////////////////////////////////////////////


//Reset ROOT and connect tree file
   gROOT->Reset();
   TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("SDataR_Neutrones_4,5-12-21_new+-12V_fondo_4.root");
   if (!f) {
      f = new TFile("SDataR_Neutrones_4,5-12-21_new+-12V_fondo_4.root");
   }
	TTree* Data_R;
    f->GetObject("Data_R",Data_R);

//Declaration of leaves types
   UShort_t        Channel;
   ULong64_t       Timestamp;
   UShort_t        Board;
   UShort_t        Energy;
   UShort_t        EnergyShort;
   UInt_t          Flags;

   // Set branch addresses.
   Data_R->SetBranchAddress("Channel",&Channel);
   Data_R->SetBranchAddress("Timestamp",&Timestamp);
   Data_R->SetBranchAddress("Board",&Board);
   Data_R->SetBranchAddress("Energy",&Energy);
   Data_R->SetBranchAddress("EnergyShort",&EnergyShort);
   Data_R->SetBranchAddress("Flags",&Flags);

//     This is the loop skeleton
//       To read only selected branches, Insert statements like:
// Data_R->SetBranchStatus("*",0);  // disable all branches
// TTreePlayer->SetBranchStatus("branchname",1);  // activate branchname

   Long64_t nentries = Data_R->GetEntries();

	ofstream fout("neutron_time_energy.dat");

   Long64_t nbytes = 0;
   for (Long64_t i=0; i<nentries;i++) {
      nbytes += Data_R->GetEntry(i);
	fout<<Channel<<"\t"<<Timestamp<<"\t"<<Energy<<endl;
   }
}
