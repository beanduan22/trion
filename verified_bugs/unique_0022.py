#!/usr/bin/env python3
"""
Bug #0022: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['branch', 'three_branch_concat'], ['fusion', 'depthwise_conv_bn_relu'], ['attention', 'matmul_4d_batch'], ['constant', 'redundant_reshape'], ['broadcast', 'tanh_add_mul_chain'], ['layout', 'resize_linear_aligncorners']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0022.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0022.onnx")
INPUT = np.frombuffer(bytes.fromhex("38a630bfac8e1cbe1e2c013fc62f4cbe7dffd43e3276153f492b883ea773253fd25d31bf7b47783fef6796bf20f5ebbd5390acbe7803b8be8f3fa13e22a0d1bf1899c0befac601be608c593f04c4933d1c80863c2f9746be0bd3b6bd7095f1bba99b0abe797f8dbdf0a3a7bea7be32bf7324b43e4dd6a33e8b9b0a3fce21aabec588533e600f423e89b3733f3c662bbfe95fb83e05b045be9ca4e73e0e9e2b3fa311953ed8f6a73e13f127bf7127333e3e3e5bbec04756bf99d9393d4c520c3eb6cffb3c1db82a3fab6f313fd8520a3fb9e7523f6349133e63cea83e66f757be441d353fcd28b4beb044213faf1ce83e00d7b1be660bd3beb5b8813fd9ac643f73b1673e91e588bf92e49bbff40662bc17b9a9bde6b081be7604013ffab2c1be631887bfc37db9be981cb1be064c8f3e4b4b76bec4d2533d43d255bfa8bbd0be51b3ae3eda16df3eefb90dbe4a7b0a3e684d7e3d0558d6be302841bde6b6d73eec23673e55e6fdbeed95083fca3094bde587993eec030c3f9728d63e7d1af1bd6e3d213fd6aac83ec0db82bfd603bf3d31f4a1bc8fb7f33b3c3ee93a8ba217bfc2a418bf9f25e6bd5e5a973d404bc33e003f9b3e8f6a7abedca14fbf57fd2dbddcd5473f52dd183f647a0b3f209e1e3f5b7305be83a50fbf854a0b3fde640b3f41a65f3e6afe36bf30fdccbd2594b13d3e0336be166154bf191da63ea94a18bf1085d7bcf0d9933e7bbe4d3f67e09b3e80dd40bed775563f4b0569bfc2b28b3e3287c03ea6dd29be2d5a253f325970bfd3bcc4befcd233bef7d7223ed15b173f5744aa3d1c9236bf00850e3f0a954f3efae7493ec461b83e7f799bbe360a1abd6e12023fdd94813c2f671a3f3297873e3081973ee43771bdfa92ef3e7df393be2536023fea067b3f7b92fcbd3271253e0009503fcc77fb3cce01023fa294d93d45c0f43ecfee033f44ef613f2fe9f3beec1f1a3f0ba3383e362d443f5659afbec0fd65bd5fc0033f82a8c0be804aeb3c6de3943ffb07b53d745cbbbfdf07f9be62cd0dbf179dd13e580646bdcdac203f9d7c7a3f2ad827bf128d91bf83f899bf67da0abf5308e2bedf8bc3be65fcccbeff2d34bf6fb5233f3905dcbc496e4dbee716aabde9e004bf6414b6bdc0be26be18f69a3fc3a3a63e172df03e2bcd343fc3df62be4fde283f2080583ee77f8cbe382d183fc2a151be4f5136bf512e9c3e7cf9d83e9cfe95bee7d5f43ee103113fae29c6be7702523c0a687e3e27a115bcd1b16ebec11bfb3edbe08a3e2956a4be5ffb82bd30fe69be78bf0b3f3399123f559bed3e8b683e3faae51abe3317b2bc1503bd3e106edb3db63b283e2444d5bde4044dbe7c8f04bf7ebcb6be44ec8c3fadc79d3dce8641bf6f81e1bd84108dbe2a90fb3ef62989bdb31b39be22074abf2fb8d13ce2e0abbee0b310bc6cc356be51d5543fe9b1a0be5317eabed6d56abeedaaa6beaaba46bf573b533d8a03d23e8b351bbfd801c3be57ba33be886ffb3eea455e3fbe2c8bbe87aa0bbf663d7b3f683fc83ececd6dbc8e778c3e56eed0be420dda3d0682713e85151ebec19103bff19e773f4d3e95be47c8f53e79c2913f761503bf0f8b873fc218f43e1e8a64bf853c393daed5af3e707bc9be9395a23eaf9c0f3f222509bfcebc35bf031cbc3e1f6ce9bb1422d43e268f873efd18c63d7f8c3fbe514accbea79f173d9fc932bf10997cbcec2f863fc983cdbddf9e80bdeabcc9ba02de943e5e911bbcd00776bf2164743f6f88873e4896833ffdd12f3f7e4cacbe24ee2fbb12ac203f623743bf45d9b53e1fd92e3f626456bee4bb853ec835843d676047be11e0103fe87434be51db9abea27d73bece2cc13bcd6e07bd4c5b75bf8e26383fb66be6be1dfab33e75f118bf0695d5be2de40a3d7f0da33e5d79f13e7cb8a7bbc31c143ebf4e4d3e99d3013e7d17a13e871361be1a336b3f1d3d553ed9e60d3c1fa3be3efaa5573efb745cbfec4600bf371ed23df7bd8c3e3f3ccb3e6244093f544e8abe5040153c65223c3f3b620e3f1f281ebdec22cd3e5ce07d3e5dadda3edbb1793fa1e3bd3e011154bff0ef823f0559f63e33a717bfcaee3dbf310c7f3f18b9703e200d5e3ff7ce7f3ea2641a3fad032c3f56a4013ff7232dbf4cb2fb3ea1f6373ed8ad61bfc0a4debcc0cb90bde495963e7424433d56a787bfb609553f5e85953d52ea83bfaf26753fb90b1a3f117d2fbfaa45733fead22fbe63da523ec18eaf3d146ba43e690e063f23909a3dd63fddbd80e3133f038ca33e9f57503d228688be36abfa3e17c42abed24e773ef90b91bd6a5012bf5d15453ff419223f1b3ca03ebaf7703f8329a3bde5f9f63d1776d63da363bebe563acd3e2332b0bedcea37bf1a7e92be43d32c3f6f3f23bff0cc953fc075c03d90e660bea6d898be1f73043f4e596bbd2347543f94dc18bf3a512abe363b293fa993793ef89a8c3e34598cbe5ce5bd3d41a586bee9d401bf2ea138bcd3c72bbe9c8f43bf6b35803efb584dbfb71c873ea8d120bf3d61d0bd680fd03e06cec43d01d230bf1ce8613e472e2d3fda34fabce37d563ee6f0593ee7495e3e11fc813eb882f23de06d463eafef833e272a1c3e7970913e3634123f0bf21abeb0ea3f3eeb528a3ea433343de8b9723f578b253ee2bb8b3d46117abe8ea5843ea21894bd4480f3bda4df263fe43d82bed1c23cbd8bc246bf8394f7bd6412b6bd0e6eef3e1d35e53e429c2e3feec589bea047a8bfb540d33ec202a7bd202fc6be6c707a3d8c97ce3eee19cabdd6f1993effa1bf3ecf9488be72542b3e8f4cc93d209d603ed57b76be26ed02bdda6e0dbeb902e83e700b0fbf2f2656be5c3db63eb5e82bbfcf4a8b3e6f8800be2e66a93ebc7e313f22500abf223490beaf975ebd0cae8f3de5644f3f5ce16b3f29e4afbe145054bc0dfaefbe8ce184bedf7d373ec1d1e53edd51f53eebdaf73ee6836cbdf6cc75bfe666573ff0b6b5be04c7bebe9e6813bd0da52f3ecf93103e2dfb9abe94fba6bb82776dbfd2ed4ebe0345743e546c913f26a940bf706d92bdb95e623dfcb34a3f0acecebbee9781bf86bbc63d722270bf1e3032be56cf523e2e6f7abef2fffdbeebed16bedbd4a43fe8501dbf0b7ea2be446cf43ebc96323e179b46bfefca4ebed734313fb55f68bfe4af88be17d0843eb02b4c3d087602be5981333ecdaf073f799644be514369be9a61a83e68fe8b3effab9f3f1e91d7becc7d75be3198833ee7aaf8bedfc3043f355587bf7fa2323e3b52acbe2de1acbe150fc7bee129c93eb899323f2c941c3fbd4733bf9bebe23db82443bee04ca9be63c7afbe74bca4bdeb450bbf5562dcbe2f0bd33d7262013d2aaf28bf038626bfee746dbedf563bbee5a7873e96293fbe998c6a3e121904bd281d54bf4d27b4bea98390be160c45bf9ed9b13e7d7ed03d58f065bda5ea343f6179c13ed4329f3f581ec9be1d3f72bec8e791bf26b4853ebf50283e7acbc43e1b7741bff2b58bbfce4303bf418a1cbeb49f67bfa8ba013f35c45a3fe287dd3d07e3f63eec669abe6452103e9b0d233e3d9f23bfff23b53d822310bf4a2a08be421fe8be90cd293e8b1b953e83cc45bdf9a743bec2d020bf5e6d71bf9504a93dd4c8673ec00b9c3ef609483ff073b13e40c0fdbd3dcac13dd3b9b1bf561a6abee327b2be0d7c363fafd61abdcd7355be360593bed14201bf4e97393e2f19d6be8007813ee35e043ec1fe76bfa44aca3ea87f543e389bc8be586c37bd95400a3f28b89abe4184563f48e9373fc6a74d3fd8ea6abee7fd90beaffe3cbdd0d9893d40c40abe0bd2963e3660953e1c6ced3e7f28e3bddc3131bf1dbc2a3ef35c123e17b3543f0d809d3e72d93e3f1fdab8be606709bf1ab472bece7b74be0aa49a3e1efe543e6c057d3f0e829c3eed2f663f573d3dbe0682d73eb952e6be39998a3d1fc9ccbe01d0aa3e5a7c443eacb0ea3e9f938d3e8826373d0a47bd3e990bdb3e4453b8bcaa2b11bf8d7bdb3ed9ee6e3f72b6803e3d13603f29fd03bfa189863f0bbca2be0e8956bf1b838abd8bdce53ed164a33fa41148bd408835be2b6d42bf26711e3e0eacaabe7e39a43e8eea013fefd1913e20bb883eba7396bed3f2363e874cc33eda78cdbef49acc3e87a4013fa182d63e35fc5d3ef0e0ad3e3b72a03d5aca5f3ed70533bfa6d9d13e2b3b673ec5340b3e819e173f75bf383fbc94db3e588c6abde4e7863ea8a89abdaa0eecbe3cd082bffbaa643ddd113a3fe443d4bd1ab828bffb6c603fcbcf003f59081abf15f118be7fec2bbee012f3beb0a58d3fedfedebef84046bf4c67683e6a1241be39e0123f41a7813ebd49e33e0061c23e006014bfab258fbf48d7373f44b161bed20248bf0c50833f18c80abe8cb8dcbe3ce43abe6da0a63caa9c4e3dfc1f1b3d31da75be1843b13e0e48a4bfb4fd10be58adee3ea81492bd6714ce3e804d51bcab3ff73d72a957be9a47043fc610e63cf98d27bc3c0c02be174ef03e92fcccbe577f0c3ffc80a73d5b7360bef3d76fbe48d292be73d496be24ed71bebd39833c88c5113e4725c33e889af3bef59f4a3de1f3d7be37f895bda960c73e484f6c3d73535ebf9820163f655107bfb26546bea0035fbf5715c8bef11daebd5e0e6fbff9a5823e172c733f5572cdbd5352a2bf001d903eef9e973f6d503b3f2ab8babef75e213fd90e9cbeccf4ebbd783cb63e55e565bec5b185bf6dc9c6be4d12b4be7f93953eb688d43e3091dc3ec30f8dbefd724b3ed09561bf22da64be5a00a0be0619cf3ebb2b2ebfc23529bf964ea7be8d3335bf1bbf9f3ee14379bd320350bb227f353f59aa9f3ed8831a3f7d8ebcbeea8a983f9f641abeb62355be33b8a33e313e2a3f866b693f0d3b423e7a58013f623cf9bef343fdbd540dfebe4ba6803feaee48becbf5953d28f87bbb492ea33c5c9c653fab356a3faf3c03bf5da2683fbf27a73fc83f23be7125763f9311c43e9c49efbc772571be3cb506bfd2561dbfc586d63d1489a63f6dee0bbf668326bf4baffbbe179a9d3e06ce2b3f80e00ebe0c3760bf78a443beeb129dbcf75af53d901e9bbc5e41e1be0869e23e0858143e2b8950bf5f46243f9f60163f366b0fbe16637cbe0acabf3e5ce0443ec0c263bf6f586dbfbdb3b63d9433cebd02a5993e5fd1353fa5d9b53e0f7e74bf82bbaf3df881cf3ed6d6293e5afc3bbf7c8ce63db15ee9bd529e8e3d5ced0d3f3ab7193cdb063c3dcc9c9c3e808cf8be198102befb861c3eb74d41bea82706be545b943dcfdb613db7ccee3e63f271bfb781b83eb8d17a3f5dbf25bd6ae558bedbd806bf85981f3f3ab094bf790da53e1c78683e5b978c3e401335bf3c6c813e4ba60bbefbbfaabe11835abfb3a953beff00ed3ef62283bfadbb953cc8c640bf229b42bfbd2e09bfba61febd14a2433cc8b27bbd5c2833bf6fe1a53fc678263d01af89bf6d93b13e5c7228bedadc0bbff007ee3ed018273f7c4b613f12430ebf7327cb3dff23103f67b62d3e0b8905bf03c70cbead494a3f2afec4be48ae173c87f25fbd25f580be42facebe21061bbf2b1581bdfceaac3d1c31953efe3cf5bea3e2833dd15633bfce3f563e492745be564462bfdea020bd151b6c3da7ce683fb3b35cbe27ff633ebe32f3be2656a1bffcb6283f9701963e050326beaadaf1bedd2e593dec8b3f3f39bb333e48da863e6b41d0be746f1a3fe021143f968f883cb5a5ec3ce02aa53d6b7b0b3fb1b0a5bedd598ebf99bb503e1f1ca0be24af69bef287143f192e0abf15948e3e19a7a13d71192ebfbfd8253f9109063f3869b0bd18be403c79586ebe67a0bdbe764a2d3da66bf23ddd25f1be0107cabebb472bbfc09d82bebed520bdd71faa3b6c8dc23e7edf463fb06787be061c60be55ff3bbff8ee1a3eed2909be69b6f3bcf7a5b83ea5fc48bef627123f8bacedbe53676ebf992192bcdd2715bf31655cbeb16f1f3fad06993e0a4abbbed4c21dbf9e816bbee1d2113f2031c8be315e39bfa6502cbd6e542d3f529514bea8d3d03ec272a0be985a573f0ba036bec4660d3fb0bc29be7bba9c3dc6655a3ff63d843ea2376abe7f7c9f3e2eac1fbfd98fb0bea6f659bd38ff833e648e12bfbda90abe6725973e1082fc3ee53539be3ed6c03d323651be8a768cbf4298393fd391813e6b4e333f45c1acbe361cc03e443a86be3337ce3ec7393dbf8f6cfa3e3724783eac7202bf64ca603f9cdb393f755e58be8e2980bf9e0b4e3eae53983e7c1bcc3e81a823bf6f86d73dc388d9be98520f3cbc528bbe22e30fbf6547523f5043503ed2f23f3f171d7c3f5980b73df82f093e3b1bd63e2bdf96bef8c63c3d5e23023f1586ec3d7436d6bed7f417be647504bf4de7cc3e19c894be8331a6bbc825373eb4e11e3f1108bf3e53740cbecdf3963e7ec7643e8852a53e19cc4bbf64a5853fe6eaca3fdd2558bfeebb963d23166d3d81e9a73e524c333f01c4b4be820df63df2c08d3e1e948a3f4f8b073f96b09d3ff0ab53be8d3f5cbf34c7633f9d57e8bef7aca9bde4d7dabd7e8ad5be172413bf25ac093f4a83c8bd2ce74dbfad5a3f3ecfec60bf55ea213ee2cb063df1ae05be2c828b3c129ae23eb4ae823e35a79fbe625e8c3b78697e3ee9c1a33d7cd5fbbe8553193f699dc4bd95dec4bed080923e060855bd7b8b823fd678c8be48e5843e3ac62bbda8e0a33ea3bff43ec84c5cbec49fb9be1a9e72bf8018bc3d2f834f3ef15b5cbf2752dabe67c2d9be80c2813e803f96bef8a345be9875d33e7e8c353fddc4d1bd8afbdd3d7026d2be8a7ac43ca7d9eabdc36369bfa97e423fb008163f10a6ff3e80aec63d759003be774e563f2551cf3e252f233f227f223eb251e53e3701a23efd893d3e3da422bb649e9bbdb66aa3bedadcdabe3c2234becd5505bfe64dafbfe4adf13e0483ad3f846190bf92ca17bff1a9e43df33898be539d94bb147ce3bed4c2f7bed0a606becc3ef5bdc97635bf9f934cbeee1ddbbebe9ea43e53aa50bfe414e83e12f55a3f6489633ee7d7663e7a3cbe3dac7104bf3353003f96bb65bf4b720c3f92bae2bed8987c3e9c13f2bee4c347bfe7aaafbe24e4063fe18d853fa9a6243f8a490c3c47b317bf4a63d73e472eb03debd9aabe102048bfcedfc93cffd808bf270e843ee05152bd94186e3f7097a5bcee0cba3e197e033fc961d53e9f56593f4803d63c056b1f3fe69e813db96066beec8b153e1b9edb3d76f5e0be8d1b36be5d66ec3dcf82e2bdd445803e04a1d2bea4482a3e7ae7233e4b01133fbb4b243f613b2abed9ebf7bebf62aa3f3f0605bfe25e863cc91d993e91a10fbe8f721fbe164ba73fc78f0fbe85bb56bfe92f2dbdb53df03d7f28ccbe347e22bfaaa579bf395cecbedab8a1be2967953e6cc0c33ec123163f137f53bdf741e9beb312c93e3636493f96ec0abfa480ffbe64693a3ed37aa93e1e1b153e0d4b82bdaaa6653ffd6add3e4b8a95be3d96dcbcc5a25dbc86b295bdf2f7513cb189853d24e38dbe995314bff3b8f43dd3ca52bd01e4abbf7e59893ef480433efa8dba3e18f73d3e1b5d86be34b3b43e77eae6be8bf6ef3e938d32ba4d610bbfb6d9923db1cec8be6be1573ea511953ed3ed383f523d3cbee17d9b3e2f8908bf4d659fbf3ce4533e176080be483f1a3fa85190bf6efb153c0dfb1b3f09095cbe0372653ece3f3b3cee60b1bea910d5bcbe6b06befbf5b3bb65c72f3e41a0d2be7b96073d29aca3bec3ec88bdf2cb39bea004553e8295a33ccec1cb3ee22c5dbf69f6ab3fa5136a3d17b1303eb086e2be488ef93e8307c4bc3f243e3f3224ecbe718420bee0dd93be63f3053ee28b903e5f69233fb5b09a3f798307bffd4a34bd2076bd3f642edcbd55ff22bdea38723ff78bbcbee6be3c3fc241073eeab7743e4129243fb0e67b3ece59ac3e01950bbf8d203bbe39dc083ec9da5d3e37b00dbf44da77bea47e3c3fc3a323bfef58373f1c567ebee338bdbe8c7d123f15ea21bfd2b9fa3ce93fe33d3b57babf5406953ed25c0bbfae3225bdf61f49bdce79fbba6aa30dbe48cef1bea8fa1dbe8f33663fcd05a3be7af60d3e0c283a3f094c323ecc501c3f0501c6bdbc4f2d3f13118cbf4d27c9be6696193e282d5cbed618db3e000a043ee2704bbf7aafef3d26aa0abe27003a3f1c8acbbd10e6d0be04a96c3ed37e85bc6390f1be2b31303f1bea5cbeceba11bba287a43dba04a5bd677a913d62e10abf16db9abdaceb8a3f49b7babe02e4afbd8899d03e2896f5be1195643fc89ac2be952639bfb132ccbec512c83ecade3f3fc080113f0f07503f7df4343f057de8bd975a07bf4608a73f467971bf8a86c9bef8fc543e7a70c53dcc44123f17e7f1be25bbbabe8f740abfffb786be751f93be2c9d9e3e133928bf014f453e536333bfd9d4e73d4fc7ae3e0a168e3fb24ac93de208a13e34bd5f3fabaeb73e57a478be7d3b31bfe9130dbe8f3693be2d3572bc3f6b303e4e149bbd2408703d4e435abe53c2ac3e09bb45bf42da8e3e9ca1163f4b69643e01da073ee4f292be796a053f2346063f6c8885bdc67fcf3fd41c12bdb50c013fcafa0dbfc66ed23d5e9b043f6b5d3fbfe6d0f7bd6a7e1d3fa5d91dbf7b621c3e17b5843a644ea93e592b05be26f8adbe318e183f2d37db3da2e9473f601d05becc0f9abe377d43bfb433af3ecc3d843f29e2d43df0ae1bbf3073833e469acebd87fd323e9a4258be8ceb0abf896c87beb138a73eac7489bf222cbebc802522bf123912bf203032bebacc0bbf749bf7bdbc4d6f3e67ad143f228a8cbfecb19ebefde6213ed5a6b33e161239bfad9639bf1b12fcbdd3d73bbeecfc093f020dfe3d0040edbd2536733e1002053e6e14053fb9f09e3fec8b3ebe9d16fc3ebc7f843e0182d7be1f84623daed53d3ce6fe10bf236bc23e5fca1a3ef8f984bd989e53be74f1bc3e35719a3edbe383becf322b3fd5aee93c5b632bbedf808f3e51e162bf230363bfac2e25bfac816dbfaa3d033fbd57b8beec040fbf612b3dbe4717473eb31621bf938492be4172533e1f2844bf71a5623f9b3900bfa67bc73e38d6e4be107a9abe2b2fa4be4598d53d8ffafcbd512bc6bed814003fcbd77fbeac27f1bd315817bfcb363abfa67fa1be74ab0bbfa557093fe06a4f3fe028a63e12a2c6beef5ea9bd43b944be8773abbd693739bea290c03dcd13cdbecc02f03de563e73d8946e03d560e06be9b172abf19dea6bd5bb3dd3e7410303f9b119abe61a706be0a8a493e6182b83dd1dd273fa909513ed2750c3f93c7a13f04fb39bf90d7c7be94c00c3e4e60b8bdd7328f3d5fd7b1bd18b013bf1979fcbecceac23e5628993d23bacdbe1b3722bdfd3a9dbf5f5578bf6b0699bda174c23ee257533e6f51f83e3b41fa3cfdfaaabe9043b43e4fcc2a3ff30826bf04d1e43e23a7ad3e38f9c3be04d427bfcc6f133e80618f3e12f996bee5dd743e75d0f83ebbffcabea12ca83ee638373f9a38f3be5adec9bd80e7ef3e9154f2be0c3dc6bdbc1a563cf43381bf5d4aa5becf1c783e036b91be1645a0be931452be1fca5fbf899338bfeca91c3ebdd01c3fd029013f0b4661bf61752e3eff00673f6647c6bed356e03e0353a4bf5d05cc3e5fc5d43ed7b8643e4a0de43e0ade913e5e6f123f4a55093e6913863fda0b2e3f822e893e5d338a3f226426bf480a42be3c18a03e4e7cd9beb859db3d0062d33e5b4fd03e34f5c53ea1b8c0be429a213f3ee0763e650b5b3fcc2d81be5582383e4f9e5fbd16a1d03edf3c0cbe7af5e83db915ce3d47908dbc511185be4a5b193f0648d83db682b13e1039823eb3d311be42e516be8db880bf3cb3cabd2c7fde3e6fea87beb90b953dea66fb3d57f0173f6c7f9d3e0cb5efbe6d81c73ee8822dbca25d803ee0e9683fa94f243e031043bdf9bfda3e32b8cc3ef21ba13e87d203bf3d8308bff7bc093fd9a8543f627116bac62d063dab6ffe3e287f263dc7d5d1be628c12be71ffa8be6af38ebf0710fa3efb9c96be43459a3ed185b9becd307cbeacb159bf856c293e9551ee3e7742523e77bd00bf2897aa3e2e4d42bef6fa213fc16ed9be861167be02639cbd2e2694be6d319d3e565b52bf0d2474be2ea298bf1350c9be26dc373fc819023f968e983d7e3da33e68215ebe4918243ec583b83d849c3abfc70e34bf8762633f9bea15bf48cf35bde0570cbffe4608be8ebd783d7e3e463e47c117bd304b813ff288cdbee3ab803f4b585b3eddaf1b3f3c32543f058c89bee407793a376a4b3f1dea65bf77cd003f8e140abeab2809bf6f2834bf7a208d3e67319c3e289c6b3f47f938bfbe89043f515c39be457a9b3e07d98c3ec40fe43de1d230bf5c617cbe6e89543e65c3ce3edbe2113fde6040be1062033f4c4b413e912c6d3e9bdcec3e98d3e53d00cc5ebe4496f3bcab1d143f3129923f8deb423e7fc0213fc6263abff09bb13e4fbb7d3f01af083d8d0f213ecf0aa5bd9c1620beda3c853f306d663d01b6b73e7301cdbec3200fbf0bfc5d3f19c9573f3bedc7bea74d50bfd70f09bda6cc4a3e92c4f93d2a4a283f5d34433f54f244be11f2fbbe0af5693f47b8d9be35de583e6d12d4be00ef0e3e0c4cf23dee8585be2e2b03bf70a71c3f99fe193f1a05adbf80bf16bf9adfa83eb12e87beb658c0bd27b689be7111d5bee5810bbfa647d33cec1fffbe07940dbf05f549bfa17f003f456161bf67d9a4bd4d221dbeb8cd823f754ba93ec2be10bf6fe8043f0695183e8ef75dbecc361f3e74a5b5be9f05023fdf4d78bdfbecfabe9da95cbe7f1e77bd4763363f75a2bd3dea40b5bd3896003e30375fbf4d5fad3da4dbd2bdcd7e163d5190a0be17207e3e213e5cbf807daa3d40d36abeace0b8be434a66bec2d45f3ea86b833c893cbf3ed9758bbea57ee1bdac84ea3e8dec3fbe0d1ab6be2f15b9bec9194abfa5ba463f2c72d2be57946fbecc41533f3632d23dedc3813cd5365a3fbb4ec33c21fb1e3fe973e6bc4a9482be7391dcbee9ed343f663888bf8124643ece7eda3eb0c302be87e5f33d03576cbf6d45db3ea2c4823e4cae05be5900513de84c223fdf6c4f3fcf70483fcb5c75be664d9dbe5b00043e4ce6263e8ca39cbf9db2ea3c276d50bf84d104bfc6a19a3e4c9329be0b51863e1f5296be1fee1cbef8c86d3fd37d6ebfaba5c43e1636243d8b2484be858b20be8a0cfb3da55d503e30f1eebe237bb73e8cc25e3f6083813f478514bee11352bedeaf4a3f98fdeebd26e4ffbe50ecd73e8ce0aa3ea6736c3d0787683ca811ccbd5b1f6abd6a48603eed0f213d0d04073fdfc318bf446504be378c98beb800a1bf7f4d0e3f7e2a653e01dfdbbda309f8bebdfc6c3f179810beeb45a7be982316bf3d50fabe31590b3fe3dfc03df037e7bd047625bf1f3e13bf51ead13e3e2f803ea0d105bec8bae5bd4ef80b3f237dc2be11d9293e451fd7bc17f8073f36bc903e1aa43c3b88b698be20cbc33ea571303ef08df2be66dac93d28b458be424b60be6548443f72fde63e997882be8cd014bdb4345c3f9ff9d93d04f3bfbf734edbbe1c76583f82600e3fd7fdf3bc47dd493fa18dc03dd441863c31e4e4be62c7173e88fbb93e2f20b53e93620ebc65e671bf51f7b13e4265c1bef95800bf9a234dbd97b39abef095483e01d68c3e77961e3f084e433ff05801be13d45ebdef932abe13b3ba3d084b98bf22ce413e6495463f888cce3e356087bff5ec9dbef0f852bdbbb4943e4ee85a3f865ef63e5db012bf8646b63eac66003fa7f0733e634fbcbfce4a993d01e4dcbedec13b3f92aee83d268e263cc36e823ebb15e9be161312bdfe359ebf6fd542bfc03f3c3eecefe03da0b4c53ebe0112bf728dabbe2cf2b23ed85be73e2e9d323dbfd62dbf86f7a0beb6cd8dbebb047c3f5af20c3fce6d88be4ac027bff530853e3787e4be8d72c1be75a70abf9198833ec540e9be1264cabd526c1bbe09d99e3f91e063be37ba1b3f8f3dc9be57b3bc3ec1cd2cbfbb631cbf2e976c3f37ee3c3eb8e9e7bc9a5851be1ff90ebe5eae7abf2c3618bf63c7b53e777911bfff806b3f6d01393e52fcfdbeb400d53d624cb8bdc78d1cbf64400ebfde50a43e574a3a3dbc92903e7548ad3cedfbccbeeeaddcbda51554bf6b30ba3bab8f423e5109ee3e4b15573e34ef6bbe21d3b03e21ef193fbf2e013f974e6ebf5c06253e67d4783da53ca2beb05cb4bcdff53f3f4ec2da3e627c863c75c8f23e688a91bf05f46abf6e47dd3eeb212ebe1638013f1dbfc93ca55c313fc1906d3f79db67be3caafc3e42030bbf1a50893e437ed1be96a4ab3f56b5e7be524f7a3e81ad103f8e1bd3bd72ddab3efc9a11bf22685bbe9f91823ecf04923cefd2d73ef9a417bf74fab8be32b66cbe36772f3e65bc62bfdca42dbfd321483d515a253e1acc2ebf876f5ebd8968b0be84879fbe24cf9c3ee38f0a3e31015fbe72b753bd57bcbabedf2473be3120a63e8611c23d97ca2cbd831e7b3e37c184bfba6762beed46d1bebe73353ed19d2b3ffb2a45bd8e0b92be93773d3e112c873ed77608bf3a395ebd1aa27abdd41d3cbe3c54ab3ec8dc5bbe1f7420bf832f45bf8c4a42bf415722be32a0ce3dd310febecfeb42bfdaf9943e6a974a3f9a0e50bda529c5be0ade1d3ecbc093becc21d73c9bb1d9bdea60713ea8d76b3e90bce2bd95265bbfe22947bf87aee4be19dca5bece5b183fbaebf73ca5cfc1be88d541befcdafc3e839595bfdd9e323edde23a3db332093e26ca483f9a9847be7e46693f852ba93ed0ead23e2d72dbbdfce1aabd4bd70cbd1f8c02bf049ce83ed6fab13e2156983be9464abe7f03b83edc4bc0be7a0fbcbe1b4bf8be8880de3ce9b8e9be7dfe6c3edcee453e989f433ed19f083f2872b6bdd20681bf3ae8a83f64e9e53e2c4cbb3e6af8f7be7a5829bf0c0d83bf4d2983bd4097f4bd032b2fbfca58c23c86cc943da9dafdbd6a17a93ebbd7d4bda39463bfcb4ed7bef9cbc13ea63a7e3f161af9bd30db8dbecd2e5e3ef23cf7bee886843d02e4943d5d8e063f3c3d74be00fc33bf8c1f51beec0d943e7e9bc0bf91ee5b3e38f663bfeb9e143f028f18bfbc73f1bd8e07063f900d10bf4e8124beb4884cbee75ca0be595119be3df4b83e84d8863f7381d53e9391b0bf6f24f5bedb8274bef9456ebfa8b60a3c5d2a743dfbe2e23e76cd2bbf56bc14bf01909fbeecf336bd23f124bed4ec063da38c6fbfc7b82c3eab9e8ebe2b5339be1f788bbe6b29b23d2ddc263f7aa386be183419bee8dfbdbd2e3983bd71aad5bee4de433f767a1dbfb382073fdfa7efbe9814423b777e193e9e70eebe097a37bfcd9b023fb643823f6e45a13e5f84723eb303d9be8980043f12caef3eb1f8293ee556aabf62eec93e4f869cbe478b983e673c173f2f6f8a3e1089d3bd49177abd5a2bb83c4e8b76bf6db7213d5834163f898352bf85cfd9bb4a79abbdeb12813fde6723bf4ba8433d37ce1c3db15c34bf08859bbda9aafd3ee44330bcb616dd3ed36d003f7cf4153fae01ae3d9651873f1ffcc83b6f9dc3be423b44bee2700bbfbed4b6be6869623ef38edcbdaaed84bfceefed3eb164b3bef057d0bb487af23e95436b3fc056533eb973ff3e8560753eda98f33d0ee4a43c1aa5203ffcbce0bd2ede813de1f229be615496bd687d103f88e5f0be61c011bfc53b9cbea254fbbe44e216bf00aced3eb0fdefbdc48c7abf34b4c73e99af1dbf84ed163e90d24d3eeff1acbc4b82583e5d691e3f38366a3ec6dbc63b6e4c2b3ff0c7dabe39bcce3b214963bd0d9f7d3fe5ae9c3e8b100cbf9305dd3eb9c7fabd3d80edbe3da0fdbe0bbc443e88bf17bf842394bee289e33e72343f3ec71981beb003423ce2d317beaa2e29bfb38bd23ed7c2a3bdcd0cbfbeb33ee63dbf99423e7f69803dbde21dbe9ddf993e96f7c23e6208613c036589be388a7c3fcc64533e734531beac97233f0f5dcdbe50b6693e4a71a73d94e9e73e3b30333ec28fb73ff9103cbf6ddbda3ecac663bffdc4d7be6389633f53eac53e3174d7be597553be0bd3333f1579f8bc3adf35be426b31be925bf03b8064e1bd83857cbf2fca88be24768dbf8473833d92e39e3e77290dbf5c8599bf9b0b113f6265033fb250babecefa91bf9c34acbf4b1bc3bea865ab3efc4a283f460516bf733a723e3fe606bf6ab937bea6dc99be9ca20bbfd40867bf234d3c3edb3ecdb9a94e283fc857cc3c8b42983e77ff48beba33293e39b6483c60b115bfb245ca3e1db29d3eaa85dc3cf19ffc3ecfaabc3e6460d8be32de54bff4e2863e1c78e63ee259dc3e7c329bbe05c24ebefe27783e4015e0bd5cfb3a3ef81e57bf534e573d12cfd9bdbae21e3f63e4e5be0bd3debe2b62423fb8290cbe59f3033febe5ba3db9fdb8bd097b4f3e36ed713ed3b11c3fa26a013f55a756bea28c273fbe19ce3e08760fbe57c2443d4619dfbcc26b48bf91c52fbf954b96be6246873e82e3a4bed65430be0e858fbe66ba023f21fdd3bd2ad0813d49a3db3eb7fcbebe6b55a63ea5c93f3dad6bfabb846078bed89759bf6e9029bfedfa0cbfaa95683e35f322bfa766f33df195423ea6e96bbe48419dbe6162a93df08f6dbeed6afe3dd23bbd3decc39c3f80f735bf72fb163edc39d0be39fb473d2532533ec4bf1d3eed29aa3eab9d803f2905013fbb4e2dbfdac8a43eb51d29bf0995d23e003f49be6c0e903f2ada833e65f08abec33fbb3e869b88bee8e3a5be3285a5bed88d53be976a08bfce5b5ebf1ae0823c8f54a13e65d0083f260605bfbcbed13ea85dc93d138f22bf531c323dd541443ed1559abc2e34d2bceba787bfc232f83ec6e5f4be6e8bb7be9bd14b3fe576293edd058bbe5438a73d771513bfc5e8c33db21298bfaf8500be996f9d3eb784403eee7223be46d2153fa96b1abfef6e38bd1a63c5bd2358e7bb5832243fea2ca4be7819b2be6bdb8c3eacf65cbd0a5cd03c11e9e43e3e72d5bec1fbf9bc885417bfa140133c3a06da3ea54ca0be26d96c3fd746e33d98325dbee410453d8515ffbe96778f3ef799a73efc14e03e602934bfdbc09bbfc9140b3dc5db3f3e306d5eb9715eb9bef313ea3db945d23ec92e2dbf1cdd5c3f017041be7a5d133f2d7120bf5e155cbf1f950bbf462aaa3cab3a843fdf4f4abefc52a5be68beddbe5295be3f05901cbfe3b403bf48e7173f31f222bf107e8d3fb7a141bffb5028bffedc3d3f814feb3e17b218bc316815be099ca33ec59acabe10f968bef83e57be1454553d2da3e0be2187143fc62721bf4bcabdbe90d3593cdd654ebea0742fbcd48508be21e8323f98331f3fa171913ea2d7b73cba1a0abeb29afdbd132d623e64f796bdd9ead23e5cb2043e41e91bbff817f93edc5c02bfad53423eb43168be44083ebeb5682cbe95e383be64d7a3be407990beefc1a13e1855213f2c93453fa488debea9415bbdccf847bce224323f404e253dcde1ad3e97541fbf457b6dbede6d1dbfaa4caebe6d596c3e5984413f7604c03e7e4fcb3df19c8f3e0261b5bf9efc193f05c6d9be055ccebe440ac93ea3ee5abfc851c53ead31913e56c3873d9522e2bdf9670a3f6b83003f40be13bfc1790b3f5a2771be6144233c4ecf073ed24f8a3ef991d33ec20685be78814e3e4f40003e2071bbbe5d67a9bd970812bf2a95c8beeb53e8bb4ea2143d1ba88fbda41cbb3e77fdad3c437a453ec3fa793df0ce4dbd3f9b1fbfa7e02abdbd4e84bd2e63d6bef73a9bbf5099d83e772a04bfedcb48bf30c1e3bea901bb3f3a709d3e61022abde163703f9207a93de7bddfbe1e8ceabc9637253ffa5e40bfc7379c3f5f9d6e3ec492973f24c2a3bdb42594be9cf24b3f547250bf476fc03eb4bf9b3e188a933eb5e3fa3ed3b113bfe79f2a3f9fd1e43c67f065bfaba51dbf3db247be9b510bbefccb143f5fb48b3df9291bbf3323a63e87950f3f1164d1be2597823e00ef093f5118373e7397453e4a2c3cbe9ee10bbf5dce9ebe4311543fe8f39d3fa41d7dbe825638be0ddb08bf7555e7bd9084ddbd1f9c263fab5e60bfe02a6c3cbf5a023f0f4a813dc30994be04d780be9815633eee2524bfaa40b3beb9fb51bedea78b3ef99a76be123383be6233433dad2901bf934333bfef2b993ea37d6e3ecf87483e46ea20bf6a6d92bedf593d3fcb81e1be631420bedc975c3fbf2996bef09d8f3e70fd613e4f072bbd5d7b61bf0e9d5f3dd944113fded714be6eefb8bea008823f7f16523ec580573f5a26cd3e7edba53edc3dd03df2f4f4bec57501bf759498be78bc883eb30b8a3dc7562fbec75e72be611de4be2b403cbf2041f3bebdfeea3d64a8983e631fdf3e2d16823ed77ddfbdd84089bd10e8093d57ec1fbfd362d23ea3716e3f819690be0212af3e579c063fdadcc5bd22d0163e977477bfc22430bf066a973ec45c03bf6b25fd3e0a4a593eb2af503ca7566bbe6cd2fc3e771682bee8cd2d3f55f2b53de22ee8bd6ac0c4be03f00e3f16540fbe31052e3eb3fbe33ecefb133f31b24d3ecdc85f3e9665163f912b9a3ef49b783ebcbb77bfb95f463f9aa2153fcaffca3e0280f4be39a3353f35d1713fc06c4bbefe38693ef24540be5e5d4a3ea846c03e33a2cabe03a02d3fd3ba833de158c93ec1979bbea7566ebf707ebcbd164d173e1acb643f9ca74cbe800b6dbfa41814be782c40be0cf3d73e4f1a803eb94118bf214a4bbf93d72c3ee808353d7e2ebcbef022b6bd8c915abf27298c3ee8b210bf4460adbfaad3fa3ef504b33ea56707bfd6f94dbfabce513e47b83c3f823b783fafea8c3ed655b4bdb71c96bfd762df3e8d1e63be2dd3f7beee01203f4d4e32befe264b3c15bfda3dc51cfd3ed5790dbe9e41263eeefe30bf63f98b3e18af993f57538e3e67d203bfaf9ee3befe571b3ee9ea51be896ca13e8b74bf3ec7bfb8be4978283e9fcb01bf8590a03de0e09cbec2475cbf4eb41abf3fd2d9be6a7e51bcd594e13ec4d8233ed78459bfc6fc393d414596be6bda2a3fc7230ebf0cf9e0bed9ab07bf34b6d13dafe8c73ec02b27bf06017f3f7d59303f0203c43ea6d5eb3e368b093f5cd60cbd16735fbdfefc30bf58f447bf9b4ac3be491a023b6303f9be54d315bf844cb43ef9b6f83ec28f4a3f7e688abc5cf4b03d6a303dbf6fa9043f9042323f9beb7abdb9691f3e52d848be1a45a5be456445becb5d8dbde2c78fbdb83dbdbdd2702fbe1c428dbe90021fbff31cc7befce3763e28a858bea59b303e3740a7bdf7fc5bbe402bc0bd0d6d13bf80ade53e4865b93dd32ba03c237fa13cd2f37f3e1166a4bbd562e13d2c5813bfab73b83ed0d6013f83dacabe4ad4de3d2e5988be54d74dbf3afc8abe"), dtype=np.float32).reshape([1, 3, 32, 32])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
