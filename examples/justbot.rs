use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

// const SCALE: i32 = 400;
// const QA: i16 = 255;
// const QB: i16 = 64;
const NUM_OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1, 
    2, 2, 2, 2,
    3, 3, 3, 3, 
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3, 
    3, 3, 3, 3,
    3, 3, 3, 3,
];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn main() {
    let dataset_path = "/home/hasan/Chess-Engine/bullet/data/combined.vf";
    let net_id = "725b4ib-1024";

    // hyperparams to fiddle with
    let hl_size = 1024;
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let superbatches = 640;
    let initial_wdl = 0.0;
    let final_wdl = 1.0;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            // merge in the factoriser weights
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            // we want to save output-bucketed weights in a format
            // that is suitable for fast cpu inference
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", 2 * hl_size, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::LinearWDL { start: initial_wdl, end: final_wdl },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 128,
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    // loading from a Viriformat binpack
    let data_loader = {
        use loader::viribinpack::{ViriBinpackLoader, ViriFilter};

        let file_path = dataset_path;
        let buffer_size_mb = 1024;
        let threads = 8;

        // The `viriformat` crate exposes a useful `Filter` of its own, but you can also
        // use a custom function like for SF binpacks with `ViriFilter::custom(function)`
        #[path = "advanced/filter.rs"]
        mod advanced_filter;
        let filter = ViriFilter::Custom(advanced_filter::should_keep);

        ViriBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };
    trainer.run(&schedule, &settings, &data_loader);
}
