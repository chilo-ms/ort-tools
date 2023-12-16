""""""""""""""""""""""""""""
" VIM mapping key
""""""""""""""""""""""""""""
" 
" ■ Shift: <S-...> 
" 	■Example: Shift s 
" 		<S-s>
" ■ Ctrl: <C-...> 
" ■ Alt and Meta are the same: <A-...> <M-...> 
"
" ■ <F1> through <F12>
" ■ Enter: <CR> or <Enter> or <Return> 
" ■ Space bar: <Space> 
" ■ Esc key: <Esc> 
" ■ bar ( | ), separate a mapping command from the next EX command: <BAR>
"

""""""""""""""""""""""""""""
" stop using the arrow keys
""""""""""""""""""""""""""""
map <up> <nop>
map <down> <nop>
map <left> <nop>
map <right> <nop>




"""""""""""""""""""""""""""""
" General setting
"""""""""""""""""""""""""""""
set number
set vb
set cursorline
syntax on
set hlsearch
set ruler
set showmode
set modeline
set autoindent
set showmatch
set noswapfile
set t_Co=256 
"comment color
highlight Comment ctermfg=cyan 
"search string color
highlight Search term=reverse ctermbg=4 ctermfg=7
set ignorecase
set smartcase
filetype plugin on

""""""""""""""""""""""""
" tab
""""""""""""""""""""""""
set tabstop=4		" Most OS and software consider the length of tab as 8 chars 
set shiftwidth=4	" the shift width when pressing '>>' 
set softtabstop=4	" replace tab with x chars, but if you press tab twice it would be a real tab
set expandtab		" expand tab with space
"set noexpandtab



" http://www.study-area.org/tips/vim/Vim-9.html

"""""""""""""""""""""""""
" showmarks
"""""""""""""""""""""""""
let g:showmarks_include='abcdefghijklmnopqrstuvwxyz' . 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
let g:showmarks_ignore_type="h"
let g:showmarks_textlower="\t"
let g:showmarks_textupper="\t"
let g:showmarks_textother="\t"
let g:showmarks_auto_toggle = 0
nnoremap <silent> mo :ShowMarksOn<CR>
nnoremap <silent> mt :ShowMarksToggle<CR>

hi ShowMarksHLl ctermfg=red   ctermbg=black
hi ShowMarksHLu ctermfg=green ctermbg=black
hi ShowMarksHLo ctermfg=red   ctermbg=black
hi ShowMarksHLm ctermfg=red   ctermbg=black

"""""""""""""""""""""""
" for wokmarks
"""""""""""""""""""""""
let g:wokmarks_do_maps = 0
let g:wokmarks_pool = "abcdefghijklmnopqrstuvwxyz"
map mm <Plug>ToggleMarkWok
map mj <Plug>NextMarkWok
map mk <Plug>PrevMarkWok
autocmd User WokmarksChange :ShowMarksOn

"""""""""""""""""""""""
" Nerdtree stuff
"""""""""""""""""""""""
nnoremap <F9> :NERDTree<CR>

"""""""""""""""""""""""
" Taglist stuff
"""""""""""""""""""""""
nnoremap <F10> :TlistToggle<CR>

"""""""""""""""""""""""
" Spell Check
"""""""""""""""""""""""
hi SpellBad term=underline cterm=underline ctermfg=red
map <F5> :set spell!<CR><BAR>:echo "Spell check: " . strpart("OffOn", 3 * &spell, 3)<CR>

" code_complete
" let g:completekey = <tab><space>   

""""""""""""""""""
" mapping
""""""""""""""""""
let mapleader=","
nmap <silent> <leader>m :nohlsearch<CR>
map <F6> :set cursorline!<CR><Bar>:echo "Highlight active cursor line: " . strpart("OffOn", 3 * &cursorline, 3)<CR>

" insert a tab character
"imap <Tab><Tab> <C-v><Tab>

""""""""""""""""""""
" Mouse
""""""""""""""""""""
set mouse=a    " if you want to copy and paste the text outside the vim, just press 'shift'
set ttymouse=xterm2
set pastetoggle=<F7>

""""""""""""""""""""
" Window
""""""""""""""""""""
" Use Ctrl+hjkl to switch between Window
nmap <C-j> <C-w>j
nmap <C-k> <C-w>k
nmap <C-h> <C-w>h
nmap <C-l> <C-w>l
nmap - <C-w>-
nmap + <C-w>+
nmap , <C-w>>
nmap . <C-w><
set splitright

" :vsplit  create a window vertically
" :split   create a window horizontally


"""""""""""""""""""""""""""""""""""""""""""""""""
" Vundle  (type :PluginInstall to install plugin)
"""""""""""""""""""""""""""""""""""""""""""""""""
set nocompatible               " be iMproved
filetype off                   " required!

" git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
" call vundle#begin('~/some/path/here')

" let Vundle manage Vundle
" required! 
Plugin 'gmarik/Vundle.vim'

" My Plugins here:
Plugin 'Lokaltog/vim-powerline'
"Plugin '2945235' 
Plugin 'tpope/vim-fugitive'
Plugin 'Lokaltog/vim-easymotion'
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
Plugin 'tpope/vim-rails.git'
" vim-scripts repos
Plugin 'L9'
Plugin 'FuzzyFinder'
" color
Plugin 'desert256.vim'
Plugin 'Impact'
Plugin 'matrix.vim'
Plugin 'vividchalk.vim'
Plugin 'ego.vim'
Plugin 'tomasr/molokai'
Plugin 'altercation/vim-colors-solarized'
"snipmate
"Plugin "MarcWeber/vim-addon-mw-utils"
"Plugin "tomtom/tlib_vim"
"Plugin "honza/snipmate-snippets"
"Plugin "garbas/vim-snipmate"
"Plugin "bootleq/ShowMarks"
Plugin 'scrooloose/nerdtree'
Plugin 'vim-scripts/taglist.vim'
"Plugin 'vim-scripts/code_complete-new-update'
Plugin 'majutsushi/tagbar'
Plugin 'clang-complete'
Plugin 'Shougo/neocomplcache'
Plugin 'ervandew/supertab'
"Plugin 'mbbill/code_complete'
"Plugin 'vim-scripts/AutoComplPop'
Plugin 'scrooloose/nerdcommenter'
Plugin 'git://git.wincent.com/command-t.git'
" ...
" DirDiff
Plugin 'vim-scripts/DirDiff.vim'
Plugin 'vcscommand.vim'

" It uses signs to indicate added, modified or removed lines based on vcs
Plugin 'mhinz/vim-signify'
" Tab list panel
Plugin 'kien/tabman.vim'
Plugin 'Townk/vim-autoclose'

" Vim xml
Plugin 'othree/xml.vim'

call vundle#end()  
filetype plugin indent on     " required!

" Brief help
" " :PluginList       - lists configured plugins
" " :PluginInstall    - installs plugins; append `!` to update or just
" 						:PluginUpdate
" " :PluginSearch foo - searches for foo; append `!` to refresh local cache
" " :PluginClean      - confirms removal of unused plugins; append `!` to
" 						auto-approve removal

""""""""""""""""""""""""""""""
" VCSCommand
""""""""""""""""""""""""""""""
nmap d <Esc>:VCSVimDiff<Enter>
filetype plugin indent on     " required!

"""""""""""""""""
" vim-powerline 
"""""""""""""""""
set t_Co=256 "Explicity tell Vim that the terminal supports 256 colors
set laststatus=2 "Always show the statusline
set encoding=utf-8
set fileencodings=utf-8,big5,euc-jp,gbk,euc-kr,utf-bom,iso8859-1
set termencoding=utf-8
set enc=utf-8
set tenc=utf8
set fenc=utf-8
"set guifont=Consolas\ for\ Powerline\
set guifont=Inconsolata\ for\ Powerline
let g:Powerline_dividers_override = [ [0x2b80], [0x003e], [0x2b82], [0x003c] ]
let g:Powerline_symbols="fancy"


" ,s  -> show the full path of the file
" ,ss -> return to Powerline status
nmap <leader>s <Esc>:set statusline=%F<CR>  
nmap <leader>ss <Esc>:set statusline=%!Pl#Statusline(0,1)<CR>      

"""""""""""
" colors
"""""""""""
colorscheme wombat256  "wombat256.vim in ~/.vim/colors/



""""""""""""""""""""""""""""""""""""""""""""""
" [ Tab Operation Mac-Mapping by Klaymen ]    
""""""""""""""""""""""""""""""""""""""""""""""
nmap q <Esc>:qall<Enter>
nmap Q <Esc>:qall!<Enter>
nmap w <Esc>:close<Enter>
nmap W <Esc>:close!<Enter>
nmap s <Esc>:write<Enter>
nmap t <Esc>:tabnew<Enter> 
nmap , <Esc>:tabprev<Enter>

"ALT+ <-
nmap OD <Esc>:tabprev<Enter>

nmap . <Esc>:tabnext<Enter>
"ALT+ ->
nmap OC <Esc>:tabnext<Enter>

nmap 1 <Esc>:tabn 1<Enter>
nmap 2 <Esc>:tabn 2<Enter>
nmap 3 <Esc>:tabn 3<Enter>
nmap 4 <Esc>:tabn 4<Enter>
nmap 5 <Esc>:tabn 5<Enter>
nmap 6 <Esc>:tabn 6<Enter>
nmap 7 <Esc>:tabn 7<Enter>
nmap 8 <Esc>:tabn 8<Enter>

"for option key in MacOS
nmap † <Esc>:tabnew<Enter> 
nmap ≥ <Esc>:tabnext<Enter>
nmap ≤ <Esc>:tabprev<Enter>




"tab completion
set completeopt=longest,menu,menuone
set wildmenu
"in ESC: (command mode), disbled auto completion next part, Cool!
set wildmode=list:longest


""""""""""""""""
" Diff stuff
""""""""""""""""
" highlight diff color
hi diffchange ctermbg=236
hi diffadd ctermbg=4
hi DiffDelete ctermfg=69 ctermbg=234
hi difftext ctermbg=3 ctermfg=0

" WinMerge-style (Alt + hjkl) mapping for vimdiff
nmap j ]c
nmap k [c

"for option key in MacOS
nmap ∆ ]c
nmap ˚ [c

" Diff mode: Used to do diffput and diffget
" Switch key mapping in Left/Right window under DiffMode
if has("autocmd")
	autocmd BufEnter,BufLeave *
		\ if &diff                                                 |
		\     if winnr() == 1                                      |
		\        nmap g do                                       |
		\        nmap ; dp                                       |
		\     else                                                 |
		\        nmap g dp                                       |
		\        nmap ; do                                       |
		\     endif                                                |
		\ endif
endif

""""""""""""""""""""""""""""""
" ctags
""""""""""""""""""""""""""""""
" Set tags path
set tags=tags,../tags,../../tags

""""""""""""""""""""""""""""""
" cscope
""""""""""""""""""""""""""""""
" init cscope hotkey
function! UseAwesomeCscope()
	let l:srcdir=(isdirectory("../src") ? '../' : './')
	try
		"set tags+=$HOME/horus/apps/tags
		exe "cs add " . l:srcdir . "cscope.out " . l:srcdir
		exe "cs add $HOME/Horus/apps/cscope.out $HOME/Horus/apps"
		"exe "cs add /home/kent.chen/cscope_ctag/Horus/cscope.out /home/kent.chen/Project/Horus/apps"
	catch /duplicate/
		silent exe "!tag_rebuild " . l:srcdir
		silent exe "cs reset"
		exe "redraw!"
		echohl Wildmenu | echo "cscope database inuse, update and re-init all connections" | echohl None
	catch /stat/
		silent exe "!tag_rebuild " . l:srcdir
		try
			exe "cs add " . l:srcdir . "cscope.out " . l:srcdir
			exe "cs add $HOME/Horus/apps/cscope.out $HOME/Horus/apps"
			"exe "cs add /home/kent.chen/cscope_ctag/Horus/cscope.out /home/kent.chen/Project/Horus/apps"
			exe "redraw!"
			echohl Wildmenu | echo "cscope file not found, exec tag_rebuild" | echohl None
		catch
			exe "redraw!"
			echohl ErrorMsg | echo "You don't have enough privilege XD, just add Horus db." | echohl None
			exe "cs add $HOME/Horus/apps/cscope.out $HOME/Horus/apps"
			"exe "cs add /home/kent.chen/cscope_ctag/Horus/cscope.out /home/kent.chen/Project/Horus/apps"
		endtry
	endtry
endfun
nnoremap <F11> <ESC>:call UseAwesomeCscope()<CR>

set csprg =/usr/bin/cscope
"set cscopequickfix=s0,c0,d0,i0,t-,e0,g0
set cscopequickfix=s-,c-,d-,i-,t-,e-
nnoremap <silent>qn :cnext<CR>
nnoremap <silent>qp :cprev<CR>
nnoremap wc       :vsplit<CR><ESC>:cs find t

" :cs find s -> find symbol
" :cs find t -> find text
"
" after find...
"
" :cw -> view the list of matches 
" :cp -> jump to previous match
" :cn -> jump to next match

""""""""""""""""""""""""""""""
" TagBar
""""""""""""""""""""""""""""""
nmap <F12>   :TagbarToggle<CR>


""""""""""""""""""""""""""""""
" Clang-Completion
""""""""""""""""""""""""""""""
"let g:clang_complete_auto=1
"let g:clang_auto_select = 1
"let g:clang_use_library=1
"let g:clang_library_path=$HOME."/usr/lib"
"let g:clang_snippets=1
"let g:clang_conceal_snippets=1
"let g:clang_periodic_quickfix=1
"let g:clang_hl_errors=1
"let g:clang_snippets_engine='snipmate'
"let g:clang_complete_copen=1

""""""""""""""""""""""""""""""
" NeoComplCache
""""""""""""""""""""""""""""""
"let g:neocomplcache_enable_at_startup = 0
"let g:neocomplcache_enable_smart_case = 1
"let g:neocomplcache_enable_camel_case_completion = 1

"" snippets expand trigger with ,,
"imap ,, <esc>a<Plug>snipMateTrigger

"map <F4> :NeoComplCacheEnable<CR>
"map ,<F4> :NeoComplCacheToggle<CR>

" most of them not documented because I'm not sure how they work
" (docs aren't good, had to do a lot of trial and error to make 
" it play nice)
let g:neocomplcache_enable_at_startup = 1
let g:neocomplcache_enable_ignore_case = 1
let g:neocomplcache_enable_smart_case = 1
let g:neocomplcache_enable_auto_select = 1
let g:neocomplcache_enable_fuzzy_completion = 1
let g:neocomplcache_enable_camel_case_completion = 1
let g:neocomplcache_enable_underbar_completion = 1
let g:neocomplcache_fuzzy_completion_start_length = 1
let g:neocomplcache_auto_completion_start_length = 1
let g:neocomplcache_manual_completion_start_length = 1
let g:neocomplcache_min_keyword_length = 1
let g:neocomplcache_min_syntax_length = 1
" complete with workds from any opened file
let g:neocomplcache_same_filetype_lists = {}
let g:neocomplcache_same_filetype_lists._ = '_'

""""""""""""""""""""""""""""""
" vim-signify
""""""""""""""""""""""""""""""

" this first setting decides in which order try to guess your current vcs
" UPDATE it to reflect your preferences, it will speed up opening files
let g:signify_vcs_list = [ 'git', 'svn' ]
" mappings to jump to changed blocks
nmap <leader>sn <plug>(signify-next-hunk)
nmap <leader>sp <plug>(signify-prev-hunk)
" nicer colors
highlight DiffAdd           cterm=bold ctermbg=none ctermfg=119
highlight DiffDelete        cterm=bold ctermbg=none ctermfg=167
highlight DiffChange        cterm=bold ctermbg=none ctermfg=227
highlight SignifySignAdd    cterm=bold ctermbg=237  ctermfg=119
highlight SignifySignDelete cterm=bold ctermbg=237  ctermfg=167
highlight SignifySignChange cterm=bold ctermbg=237  ctermfg=227

"""""""""""""""""""""""""""""
" TabMan 
""""""""""""""""""""""""""""

" mappings to toggle display, and to focus on it
let g:tabman_toggle = 'tl'
let g:tabman_focus  = 'tf'


